import util
import os

PATH_TO_DATA = os.path.join(util.PATH_TO_DATA, 'large_ds_parts', 'gen_data')

run_folders = [os.path.join(PATH_TO_DATA, f"run{i}") for i in range(101)]

if __name__ == "__main__":
    features_train_out_f, runtimes_train_out_f, commands_train_out_f = os.path.join(PATH_TO_DATA, "merged", "train",
                                                                                    "features.csv"), os.path.join(
        PATH_TO_DATA, "merged", "train", "runtimes.csv"), os.path.join(PATH_TO_DATA, "merged", "train",
                                                                       "data_commands.csv")
    features_test_out_f, runtimes_test_out_f, commands_test_out_f = os.path.join(PATH_TO_DATA, "merged", "test",
                                                                                 "features.csv"), os.path.join(
        PATH_TO_DATA, "merged", "test", "runtimes.csv"), os.path.join(PATH_TO_DATA, "merged", "test",
                                                                      "data_commands.csv")
    instances = [0, 0, 0, 0]  # Need to relabel ID's.
    for run in run_folders:
        # Find duplicate ids
        features_f, runtimes_f, commands_f = os.path.join(run, "features.csv"), os.path.join(run,
                                                                                             "runtimes.csv"), os.path.join(
            run, "data_commands.csv")
        ids = []
        duplicate_ids = []
        with open(runtimes_f, "r") as in_runtimes:
            for line in in_runtimes:
                id = line.split(" ")[0]
                if id in ids:
                    duplicate_ids.append(id)
                ids.append(id)

        old_id_to_new_id = {}
        errord_ids = []
        with open(runtimes_f, "r") as in_runtimes:
            for line in in_runtimes:
                id = line.split(" ")[0]
                if id in duplicate_ids:
                    continue
                if "ERROR" in line:
                    errord_ids.append(id)
                    continue
                assert any(util.GENERATOR_NAMES[i] in id for i in range(util.NUM_GENERATORS))
                for i in range(util.NUM_GENERATORS):
                    if util.GENERATOR_NAMES[i] in id:
                        if instances[i] >= 15000:
                            errord_ids.append(id)
                            break
                        assert id not in old_id_to_new_id
                        old_id_to_new_id[id] = f"{util.GENERATOR_NAMES[i]}_{instances[i]}"
                        instances[i] += 1
        with open(features_f, "r") as in_features, open(runtimes_f, "r") as in_runtimes, open(commands_f,
                                                                                              "r") as in_commands, open(
            features_train_out_f, "a+") as out_train_features, open(runtimes_train_out_f,
                                                                    "a+") as out_train_runtimes, open(
            commands_train_out_f,
            "a+") as out_train_commands, open(
            features_test_out_f, "a+") as out_test_features, open(runtimes_test_out_f, "a+") as out_test_runtimes, open(
            commands_test_out_f,
            "a+") as out_test_commands:
            for line in in_features:
                id = line.split(" ")[0]
                if id in errord_ids or id in duplicate_ids:
                    continue
                assert id in old_id_to_new_id
                line = line.replace(id, old_id_to_new_id[id])
                if int(old_id_to_new_id[id].split("_")[1]) >= 12000:
                    out_test_features.write(f"{line}")
                else:
                    out_train_features.write(f"{line}")
            for line in in_runtimes:
                id = line.split(" ")[0]
                if id in errord_ids or id in duplicate_ids:
                    continue
                assert id in old_id_to_new_id
                line = line.replace(id, old_id_to_new_id[id])
                if int(old_id_to_new_id[id].split("_")[1]) >= 12000:
                    out_test_runtimes.write(f"{line}")
                else:
                    out_train_runtimes.write(f"{line}")
            for line in in_commands:
                id = line.split(";")[0]
                if id in errord_ids or id in duplicate_ids:
                    continue
                assert id in old_id_to_new_id
                line = line.replace(id, old_id_to_new_id[id])
                if int(old_id_to_new_id[id].split("_")[1]) >= 12000:
                    out_test_commands.write(f"{line}")
                else:
                    out_train_commands.write(f"{line}")
            out_train_features.flush()
            out_test_features.flush()
            out_train_runtimes.flush()
            out_test_runtimes.flush()
            out_train_commands.flush()
            out_test_commands.flush()
    print(instances)
