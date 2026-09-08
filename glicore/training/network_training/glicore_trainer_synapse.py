#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from glicore.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from glicore.training.loss_functions.deep_supervision import MultipleOutputLoss2
from glicore.training.loss_functions.glicore_evidential_loss import GliCoReEvidentialLoss
from glicore.utilities.to_torch import maybe_to_torch, to_cuda
from glicore.network_architecture.synapse.glicore_synapse import GliCoRe
from glicore.network_architecture.initialization import InitWeights_He
from glicore.network_architecture.neural_network import SegmentationNetwork
from glicore.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from glicore.training.dataloading.dataset_loading import unpack_dataset
from glicore.training.network_training.Trainer_synapse import Trainer_synapse
from glicore.training.network_training.pacer_training_support import PACERTrainingMixin
from glicore.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from glicore.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from fvcore.nn import FlopCountAnalysis
from glicore.glicore_config import PAPER_TRAINING


SYNAPSE_EVALUATION_LABELS = (1, 2, 3, 4, 6, 7, 8, 11)
SYNAPSE_LABEL_NAMES = (
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "liver",
    "stomach",
    "aorta",
    "pancreas",
)
SYNAPSE_TRAIN_KEYS = (
    "img0005", "img0006", "img0007", "img0009", "img0010", "img0021",
    "img0023", "img0024", "img0026", "img0027", "img0028", "img0030",
    "img0031", "img0033", "img0034", "img0037", "img0039", "img0040",
)
SYNAPSE_VAL_KEYS = (
    "img0001", "img0002", "img0003", "img0004", "img0008", "img0022",
    "img0025", "img0029", "img0032", "img0035", "img0036", "img0038",
)


class GliCoReTrainerSynapse(PACERTrainingMixin, Trainer_synapse):
    """Paper-aligned GliCoRe trainer for Synapse."""

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = PAPER_TRAINING.epochs
        self.initial_lr = PAPER_TRAINING.initial_lr
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.load_pretrain_weight = False

        self.load_plans_file()

        self.crop_size = [64, 128, 128]
        self.input_channels = self.plans['num_modalities']
        self.num_classes = self.plans['num_classes'] + 1
        self.conv_op = nn.Conv3d

        self.embedding_dim = 192
        self.depths = [2, 2, 2, 2]
        self.num_heads = [6, 12, 24, 48]
        self.embedding_patch_size = [2, 4, 4]
        self.window_size = [4, 4, 8, 4]
        self.deep_supervision = True
        self.foreground_class_sampling_weights = None
        self._configure_pacer_support()

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.plans['plans_per_stage'][self.stage]['pool_op_kernel_sizes'] = [[2, 4, 4], [2, 2, 2], [2, 2, 2]]
            self.process_plans(self.plans)
            self._resolve_pacer_from_plans()

            self.setup_DA_params()
            if self.deep_supervision:
                ################# Here we wrap the loss for deep supervision ############
                # we need to know the number of outputs of the network
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                # mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
                # weights[~mask] = 0
                weights = weights / weights.sum()
                print(weights)
                self.ds_loss_weights = weights
                base_loss = GliCoReEvidentialLoss(
                    soft_dice_kwargs={
                        'batch_dice': self.batch_dice,
                        'smooth': 1e-5,
                        'do_bg': False,
                    },
                    num_classes=self.num_classes,
                    max_epochs=self.max_num_epochs,
                )
                self.loss = MultipleOutputLoss2(base_loss, self.ds_loss_weights)
                ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory,
                                                      self.plans['data_identifier'] + "_stage%d" % self.stage)
            seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
            seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                if self.pacer_enabled:
                    self._initialize_pacer_context_store()

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales if self.deep_supervision else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=seeds_train,
                    seeds_val=seeds_val,
                    compute_pacer_mask=self.pacer_enabled,
                    use_singleThreadedAugmenter=(
                        self.pacer_enabled
                        and self.pacer_mode == 'calibrate'
                    ),
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            if self.pacer_enabled and self.pacer_mode == 'train':
                self._load_pacer_calibration()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """

        self.network = GliCoRe(in_channels=self.input_channels,
                             out_channels=self.num_classes,
                             img_size=self.crop_size,
                             feature_size=16,
                             num_heads=4,
                             depths=[3, 3, 3, 3],
                             dims=[32, 64, 128, 256],
                             do_ds=True,
                             enable_pacer=self.pacer_enabled,
                             pacer_patch_size=tuple(int(value) for value in self.crop_size),
                             )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        # Print the network parameters & Flops
        n_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        input_res = (1, 64, 128, 128)
        input = torch.ones(()).new_empty((1, *input_res), dtype=next(self.network.parameters()).dtype,
                                         device=next(self.network.parameters()).device)
        flops = FlopCountAnalysis(self.network, input)
        print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
        try:
            print(f"MAdds: {round(flops.total() * 1e-9, 2)} G")
        except (AssertionError, RuntimeError) as error:
            print("MAdds unavailable from fvcore:", error)

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        if self.deep_supervision:
            target = target[0]
            output = output[0]
        else:
            target = target
            output = output
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        if self.pacer_enabled:
            return self._run_pacer_iteration(
                data_dict, do_backprop, run_online_evaluation
            )
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()
        self._set_evidential_epoch()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data

                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        elif int(self.fold) == 0:
            tr_keys = np.asarray(SYNAPSE_TRAIN_KEYS)
            val_keys = np.asarray(SYNAPSE_VAL_KEYS)
            available = set(self.dataset.keys())
            required = set(tr_keys) | set(val_keys)
            missing = sorted(required - available)
            if missing:
                raise RuntimeError(
                    "Fixed Synapse split is missing cases: %s" % missing
                )
            if set(tr_keys) & set(val_keys):
                raise RuntimeError("Fixed Synapse train/validation split overlaps")
            self.print_to_log_file(
                "Using fixed Synapse fold 0 split (18 train / 12 validation); "
                "splits_final.pkl is intentionally ignored."
            )
        else:
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            splits = list(kfold.split(all_keys_sorted))
            if self.fold < len(splits):
                train_idx, val_idx = splits[self.fold]
                tr_keys = all_keys_sorted[train_idx]
                val_keys = all_keys_sorted[val_idx]
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
        self._configure_synapse_class_balance()

    def _configure_synapse_class_balance(self):
        sampled_counts = OrderedDict((label, 0) for label in SYNAPSE_EVALUATION_LABELS)
        presence_counts = OrderedDict((label, 0) for label in SYNAPSE_EVALUATION_LABELS)
        for case in self.dataset_tr.values():
            properties = case.get("properties")
            if properties is None:
                properties = load_pickle(case["properties_file"])
            locations = properties.get("class_locations", {})
            for label in SYNAPSE_EVALUATION_LABELS:
                class_locations = locations.get(label, locations.get((label,), ()))
                count = len(class_locations)
                sampled_counts[label] += count
                presence_counts[label] += int(count > 0)

        positive_counts = [count for count in sampled_counts.values() if count > 0]
        if not positive_counts:
            raise RuntimeError("Synapse split contains no evaluated foreground labels")
        reference_count = float(np.median(positive_counts))
        balance = OrderedDict()
        for label, count in sampled_counts.items():
            if count <= 0:
                balance[label] = 2.5
            else:
                rarity = (reference_count / float(count)) ** 2
                balance[label] = float(np.clip(rarity, 1.0, 2.5))

        self.foreground_class_sampling_weights = {
            label: 0.25 for label in range(1, self.num_classes)
        }
        self.foreground_class_sampling_weights.update(
            {label: weight ** 2 for label, weight in balance.items()}
        )
        base_loss = getattr(getattr(self, "loss", None), "loss", None)
        if base_loss is not None and hasattr(base_loss, "set_class_weights"):
            class_weights = torch.ones(self.num_classes, dtype=torch.float32)
            for label, weight in balance.items():
                class_weights[label] = weight
            base_loss.set_class_weights(class_weights)

        self.print_to_log_file(
            "Synapse evaluated labels:",
            list(zip(SYNAPSE_EVALUATION_LABELS, SYNAPSE_LABEL_NAMES)),
        )
        self.print_to_log_file("Synapse class-location counts:", dict(sampled_counts))
        self.print_to_log_file("Synapse class presence:", dict(presence_counts))
        self.print_to_log_file("Synapse bounded class weights:", dict(balance))
        self.print_to_log_file(
            "Synapse foreground sampling weights:",
            self.foreground_class_sampling_weights,
        )

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        if self._handle_pacer_calibration_mode():
            return
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        if self.deep_supervision:
            self.network.do_ds = True
        else:
            self.network.do_ds = False
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
