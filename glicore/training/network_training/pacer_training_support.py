import gc
import hashlib
import json
import os
from os.path import isfile, join
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from glicore.training.dataloading.pacer_context import PACERContextStore
from glicore.training.loss_functions.pacer_loss import (
    pacer_occupancy_loss,
    pacer_relation_loss,
)
from glicore.glicore_config import PACER_CONFIG
from glicore.utilities.to_torch import maybe_to_torch, to_cuda


class PACERTrainingMixin:
    """Reusable PACER lifecycle for Synapse and ACDC trainers."""

    def _configure_pacer_support(self):
        self.pacer_mode = os.environ.get("PACER_MODE", "train").lower()
        if self.pacer_mode not in (
            "train",
            "prepare",
            "calibrate",
        ):
            raise ValueError("Unknown PACER_MODE: %s" % self.pacer_mode)
        self.pacer_enabled = os.environ.get("PACER_ENABLE", "1") == "1"
        self.pacer_lambda_occ = 0.10
        self.pacer_lambda_rel = 0.05
        self.pacer_context_store = None
        self.pacer_pos_weight = None
        self._pacer_iteration = 0
        self.pacer_log_interval = max(
            1, int(os.environ.get("PACER_LOG_INTERVAL", "50"))
        )

    def _resolve_pacer_from_plans(self):
        self.pacer_enabled = bool(
            self.pacer_enabled or self.plans.get("pacer_enabled", False)
        )
        if self.pacer_enabled:
            self.plans["pacer_enabled"] = True

    def _initialize_pacer_context_store(self):
        if self.data_aug_params.get("random_crop", False):
            raise RuntimeError("PACER requires random_crop=False")
        self.pacer_context_store = PACERContextStore(
            cache_dir=join(self.folder_with_preprocessed_data, "pacer_context_64"),
            dataset=self.dataset,
            spacing=self.plans["plans_per_stage"][self.stage]["current_spacing"],
            num_classes=self.num_classes,
            lru_size=int(os.environ.get("PACER_CACHE_SIZE", "64")),
        )
        selected = list(self.dataset_tr.keys()) + list(self.dataset_val.keys())
        self.pacer_context_store.require_cases(selected)
        self.pacer_pos_weight = torch.from_numpy(
            self.pacer_context_store.compute_pos_weight(self.dataset_tr.keys())
        )
        self.print_to_log_file(
            "PACER context cache:", str(self.pacer_context_store.cache_dir)
        )
        self.print_to_log_file(
            "PACER occupancy pos_weight:", self.pacer_pos_weight.tolist()
        )

    def _pacer_signature(self):
        architecture = [
            (name, tuple(parameter.shape))
            for name, parameter in self.network.named_parameters()
        ]
        payload = {
            "implementation_version": 3,
            "architecture": architecture,
            "context": self.pacer_context_store.signature,
            "spacing": list(
                self.plans["plans_per_stage"][self.stage]["current_spacing"]
            ),
            "patch_size": [int(value) for value in self.crop_size],
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @property
    def _pacer_calibration_path(self):
        return join(self.output_folder, "pacer_calibration.json")

    def _load_pacer_calibration(self):
        if not isfile(self._pacer_calibration_path):
            raise RuntimeError(
                "Missing pacer_calibration.json. Run the dataset training script "
                "with 'calibrate' before training."
            )
        with open(self._pacer_calibration_path, "r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        if calibration.get("signature") != self._pacer_signature():
            raise RuntimeError("PACER calibration signature does not match this model/cache")
        self.pacer_lambda_occ = float(calibration["lambda_occ"])
        self.pacer_lambda_rel = float(calibration["lambda_rel"])
        self.print_to_log_file(
            "PACER calibrated lambdas:",
            self.pacer_lambda_occ,
            self.pacer_lambda_rel,
        )

    def _prepare_pacer_batch(self, data_dict):
        global_batch = self.pacer_context_store.get_batch(data_dict["keys"])
        image = maybe_to_torch(global_batch["image"]).float()
        valid = maybe_to_torch(global_batch["valid_mask"]).float().unsqueeze(1)
        occupancy = maybe_to_torch(global_batch["occupancy"]).float()
        patch_center = maybe_to_torch(data_dict["patch_center"]).float()
        case_shape = maybe_to_torch(data_dict["case_shape"]).float()
        if torch.cuda.is_available():
            image = to_cuda(image)
            valid = to_cuda(valid)
            occupancy = to_cuda(occupancy)
            patch_center = to_cuda(patch_center)
            case_shape = to_cuda(case_shape)
        if self.network is not None and self.network.training:
            intensity_scale = torch.empty(
                (image.shape[0], image.shape[1], 1, 1, 1),
                device=image.device,
                dtype=image.dtype,
            ).uniform_(0.90, 1.10)
            image = image * intensity_scale
            channel_std = image.flatten(2).std(dim=2).view(
                image.shape[0], image.shape[1], 1, 1, 1
            )
            image = image + 0.01 * channel_std * torch.randn_like(image)
        return (
            torch.cat((image, valid), dim=1),
            valid,
            occupancy,
            patch_center,
            case_shape,
        )

    def _set_evidential_epoch(self):
        loss = getattr(self, "loss", None)
        base = getattr(loss, "loss", loss)
        if hasattr(base, "current_epoch"):
            base.current_epoch = int(getattr(self, "epoch", 0))

    def _compute_pacer_losses(self, data, target, data_dict):
        global_input, global_valid, global_target, patch_center, case_shape = (
            self._prepare_pacer_batch(data_dict)
        )
        output, aux = self.network.forward_training_aux(
            data,
            global_input=global_input,
            patch_center=patch_center,
            case_shape=case_shape,
        )
        main_loss = self.loss(output, target)
        occupancy_loss = pacer_occupancy_loss(
            aux["pacer_occupancy_logits"],
            global_target,
            F.adaptive_max_pool3d(global_valid, 8),
            self.pacer_pos_weight,
        )
        full_target = target[0] if isinstance(target, (tuple, list)) else target
        relation_loss, relation_stats = pacer_relation_loss(
            aux["pacer_relation_embedding"],
            full_target,
            spacing=self.plans["plans_per_stage"][self.stage]["current_spacing"],
            max_pairs=PACER_CONFIG.max_adjacent_pairs,
            margin=PACER_CONFIG.relation_margin,
            valid_mask=data_dict["pacer_valid_mask"].to(
                device=aux["pacer_relation_embedding"].device
            ),
        )
        return output, aux, main_loss, occupancy_loss, relation_loss, relation_stats

    def _prepare_calibration_sample(self, data_dict):
        sample = dict(data_dict)
        sample["keys"] = data_dict["keys"][:1]
        for key in ("patch_center", "case_shape", "pacer_valid_mask"):
            sample[key] = data_dict[key][:1]
        data = maybe_to_torch(data_dict["data"][:1])
        raw_target = data_dict["target"]
        if isinstance(raw_target, (tuple, list)):
            target = [maybe_to_torch(item[:1]) for item in raw_target]
        else:
            target = maybe_to_torch(raw_target[:1])
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        return data, target, sample

    @staticmethod
    def _gradient_norm(loss, tensors, retain_graph=True):
        tensors = list(tensors)
        gradients = torch.autograd.grad(
            loss, tensors, retain_graph=retain_graph, allow_unused=True
        )
        squared = loss.new_zeros(())
        for gradient in gradients:
            if gradient is not None:
                squared = squared + gradient.detach().float().square().sum()
        return float(torch.sqrt(squared).cpu())

    def _release_calibration_graph(self):
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _run_pacer_calibration(self):
        self.network.train()
        self._set_evidential_epoch()
        occupancy_candidates = []
        relation_candidates = []
        batches = int(os.environ.get(
            "PACER_CALIBRATION_BATCHES", str(PACER_CONFIG.calibration_batches)
        ))
        global_parameters = [
            parameter
            for name, parameter in self.network.pacer_module.global_encoder.named_parameters()
            if not name.startswith("occupancy_head")
        ]
        for index in range(batches):
            data_dict = next(self.tr_gen)
            data, target, sample = self._prepare_calibration_sample(data_dict)
            output, aux, main, occupancy, relation, _ = self._compute_pacer_losses(
                data, target, sample
            )
            main_global = self._gradient_norm(main, global_parameters)
            occupancy_global = self._gradient_norm(occupancy, global_parameters)
            main_feature = self._gradient_norm(main, [aux["feature"]])
            relation_feature = self._gradient_norm(
                relation, [aux["feature"]], retain_graph=False
            )
            if main_global > 0.0 and occupancy_global > 0.0:
                occupancy_candidates.append(
                    PACER_CONFIG.occupancy_gradient_ratio
                    * main_global / occupancy_global
                )
            if main_feature > 0.0 and relation_feature > 0.0:
                relation_candidates.append(
                    PACER_CONFIG.relation_gradient_ratio
                    * main_feature / relation_feature
                )
            del output, aux, main, occupancy, relation, data, target, sample, data_dict
            self._release_calibration_graph()
            self.print_to_log_file("PACER calibration batch %d/%d" % (index + 1, batches))
        if not occupancy_candidates or not relation_candidates:
            raise RuntimeError("PACER calibration produced zero usable gradients")
        self.pacer_lambda_occ = float(
            np.clip(np.median(occupancy_candidates), *PACER_CONFIG.occupancy_weight_clip)
        )
        self.pacer_lambda_rel = float(
            np.clip(np.median(relation_candidates), *PACER_CONFIG.relation_weight_clip)
        )
        result = {
            "signature": self._pacer_signature(),
            "lambda_occ": self.pacer_lambda_occ,
            "lambda_rel": self.pacer_lambda_rel,
            "batches": batches,
        }
        with open(self._pacer_calibration_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        self.print_to_log_file("PACER calibration saved:", self._pacer_calibration_path)


    def _run_pacer_iteration(self, data_dict, do_backprop, run_online_evaluation):
        data = maybe_to_torch(data_dict["data"])
        target = maybe_to_torch(data_dict["target"])
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        self.optimizer.zero_grad()
        self._set_evidential_epoch()
        output, aux, main, occupancy, relation, relation_stats = (
            self._compute_pacer_losses(data, target, data_dict)
        )
        ramp = min(1.0, self.epoch / float(PACER_CONFIG.ramp_epochs))
        loss = main + ramp * (
            self.pacer_lambda_occ * occupancy + self.pacer_lambda_rel * relation
        )
        if do_backprop:
            if self.fp16:
                self.amp_grad_scaler.scale(loss).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
        if run_online_evaluation:
            self.run_online_evaluation(output, target)
        self._pacer_iteration += 1
        if self._pacer_iteration % self.pacer_log_interval == 0:
            ratio = (
                aux["pacer_context_residual"].detach().float().abs().mean()
                / aux["pacer_dec1_base"].detach().float().abs().mean().clamp_min(1e-6)
            )
            self.print_to_log_file("pacer/loss_main: %.6f" % float(main.detach().cpu()))
            self.print_to_log_file("pacer/loss_occupancy: %.6f" % float(occupancy.detach().cpu()))
            self.print_to_log_file("pacer/loss_relation: %.6f" % float(relation.detach().cpu()))
            self.print_to_log_file("pacer/context_ratio: %.6f" % float(ratio.cpu()))
            self.print_to_log_file("pacer/pull_pairs: %d" % relation_stats["pull_pairs"])
            self.print_to_log_file("pacer/push_pairs: %d" % relation_stats["push_pairs"])
        return loss.detach().cpu().numpy()

    def _handle_pacer_calibration_mode(self):
        if not self.pacer_enabled:
            return False
        if self.pacer_mode == "calibrate":
            self._run_pacer_calibration()
            self.skip_post_training_validation = True
            return True
        return False
