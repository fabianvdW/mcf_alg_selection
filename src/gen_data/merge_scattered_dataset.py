import util
import os

PATH_TO_DATA = os.path.join(util.PATH_TO_DATA, 'large_ds_parts', 'gen_data')

run_folders = [os.path.join(PATH_TO_DATA, f"run{i}") for i in range(100)]

if __name__ == "__main__":
    features_out_f, runtimes_out_f, commands_out_f = os.path.join(PATH_TO_DATA, "merged", "features.csv"), os.path.join(
        PATH_TO_DATA, "merged", "runtimes.csv"), os.path.join(PATH_TO_DATA, "merged", "data_commands.csv")
    instances = [0, 0, 0, 0]  # Need to relabel ID's.
    for run in run_folders:
        old_id_to_new_id = {}
        errord_ids = []
        features_f, runtimes_f, commands_f = os.path.join(run, "features.csv"), os.path.join(run,
                                                                                             "runtimes.csv"), os.path.join(
            run, "data_commands.csv")
        with open(runtimes_f, "r") as in_runtimes:
            for line in in_runtimes:
                id = line.split(" ")[0]
                if "ERROR" in line:
                    errord_ids.append(id)
                    continue
                assert any(util.GENERATOR_NAMES[i] in id for i in range(util.NUM_GENERATORS))
                for i in range(util.NUM_GENERATORS):
                    if util.GENERATOR_NAMES[i] in id:
                        if instances[i] >= 15000:
                            errord_ids.append(id)
                            break
                        old_id_to_new_id[id] = f"{util.GENERATOR_NAMES[i]}_{instances[i]}"
                        instances[i] += 1
        with open(features_f, "r") as in_features, open(runtimes_f, "r") as in_runtimes, open(commands_f,
                                                                                              "r") as in_commands, open(
            features_out_f, "a+") as out_features, open(runtimes_out_f, "a+") as out_runtimes, open(commands_out_f,
                                                                                                    "a+") as out_commands:
            for line in in_features:
                id = line.split(" ")[0]
                if id in errord_ids:
                    continue
                assert id in old_id_to_new_id
                line = line.replace(id, old_id_to_new_id[id])
                out_features.write(f"{line}")
            for line in in_runtimes:
                id = line.split(" ")[0]
                if id in errord_ids:
                    continue
                assert id in old_id_to_new_id
                line = line.replace(id, old_id_to_new_id[id])
                out_runtimes.write(f"{line}")
            for line in in_commands:
                id = line.split(";")[0]
                if id in errord_ids:
                    continue
                assert id in old_id_to_new_id
                line = line.replace(id, old_id_to_new_id[id])
                out_commands.write(f"{line}")
    print(instances)