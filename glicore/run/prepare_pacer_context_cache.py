import argparse
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import load_pickle

from glicore.run.default_configuration import get_default_configuration
from glicore.training.dataloading.pacer_context import PACERContextStore
from glicore.training.dataloading.dataset_loading import load_dataset
from glicore.utilities.task_name_id_conversion import convert_id_to_task_name


def main(cache_name="pacer_context_64"):
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("trainer")
    parser.add_argument("task", type=int)
    parser.add_argument("fold")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_name = convert_id_to_task_name(args.task)
    (
        plans_file,
        output_folder,
        dataset_directory,
        batch_dice,
        stage,
        trainer_class,
    ) = get_default_configuration(
        args.network, task_name, args.trainer
    )
    plans = load_pickle(plans_file)
    folder = Path(dataset_directory) / (
        plans["data_identifier"] + "_stage%d" % stage
    )
    dataset = load_dataset(str(folder))
    fold = int(args.fold)
    trainer = trainer_class(
        plans_file,
        fold,
        output_folder=output_folder,
        dataset_directory=dataset_directory,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=False,
        deterministic=True,
        fp16=False,
    )
    trainer.folder_with_preprocessed_data = str(folder)
    trainer.dataset = dataset
    trainer.do_split()
    selected_keys = tuple(
        sorted(
            {str(key) for key in trainer.dataset_tr.keys()}
            | {str(key) for key in trainer.dataset_val.keys()}
        )
    )
    selected_dataset = {key: dataset[key] for key in selected_keys}
    spacing = plans["plans_per_stage"][stage]["current_spacing"]
    store = PACERContextStore(
        cache_dir=folder / cache_name,
        dataset=selected_dataset,
        spacing=spacing,
        num_classes=plans["num_classes"] + 1,
    )

    expected_files = {
        store._safe_key(key) + ".npz" for key in selected_keys
    }
    stale_files = [
        path for path in store.cache_dir.glob("*.npz")
        if path.name not in expected_files
    ]
    for path in stale_files:
        path.unlink()
    if stale_files:
        print("PACER removed stale cache files: %d" % len(stale_files))

    counts = {"created": 0, "reused": 0, "failed": 0}
    for index, key in enumerate(selected_keys, start=1):
        try:
            state = store.prepare_case(key, overwrite=args.overwrite)
            counts[state] += 1
        except Exception as error:
            counts["failed"] += 1
            print("failed %s: %s" % (key, error))
        if index % 25 == 0 or index == len(selected_keys):
            print("PACER context cache: %d/%d" % (index, len(selected_keys)))

    print(
        "PACER context cache complete: created=%d reused=%d failed=%d"
        % (counts["created"], counts["reused"], counts["failed"])
    )
    final_count = len(list(store.cache_dir.glob("*.npz")))
    if final_count != len(selected_keys):
        raise RuntimeError(
            "PACER cache count mismatch: expected=%d actual=%d"
            % (len(selected_keys), final_count)
        )
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
