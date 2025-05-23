import awkward as ak
import os
import matplotlib.pyplot as plt

cache_path = "/afs/cern.ch/user/l/lmarkus/Phase2-L1MenuTools/cache/V44nano_GT"
obj_dict = {
    "electrons": [
        ["V44nano_GT_DYLL_M50_L1GTtkElectron.parquet", "V44nano_GT_DYLL_M50_L1tkElectron.parquet"],
        ["V44nano_GT_MinBias_L1GTtkElectron.parquet", "V44nano_GT_MinBias_L1tkElectron.parquet"],
        ["V44nano_GT_MinBias_L1GTtkElectron.parquet", "V44nano_GT_MinBias_L1tkElectron.parquet"]
    ],
    "jets": [
        ["V44nano_GT_MinBias_L1GTsc4Jet.parquet", "V44nano_GT_MinBias_L1puppiJetSC4.parquet"], 
        ["V44nano_GT_MinBias_L1GTsc8Jet.parquet", "V44nano_GT_MinBias_L1puppiJetSC8.parquet"], 
        ["V44nano_GT_MinBias_L1GTscJetSum.parquet","V44nano_GT_MinBias_L1puppiJetSC4sums.parquet"],
        ["V44nano_GT_TT_L1GTsc4Jet.parquet", "V44nano_GT_TT_L1puppiJetSC4.parquet"],
        ["V44nano_GT_TT_L1GTsc8Jet.parquet", "V44nano_GT_TT_L1puppiJetSC8.parquet"],
        ["V44nano_GT_TT_L1GTscJetSum.parquet", "V44nano_GT_TT_L1puppiJetSC4sums.parquet"],
    ],
    "muons": [
        ["V44nano_GT_DYLL_M50_L1GTgmtDispMuon.parquet", "V44nano_GT_DYLL_M50_L1gmtDispMuon.parquet"], 
        ["V44nano_GT_DYLL_M50_L1GTgmtMuon.parquet", "V44nano_GT_DYLL_M50_L1gmtMuon.parquet"], 
        ["V44nano_GT_DYLL_M50_L1GTgmtTkMuon.parquet","V44nano_GT_DYLL_M50_L1gmtTkMuon.parquet"],
        ["V44nano_GT_MinBias_L1GTgmtDispMuon.parquet","V44nano_GT_MinBias_L1gmtDispMuon.parquet"],
        ["V44nano_GT_MinBias_L1GTgmtMuon.parquet", "V44nano_GT_MinBias_L1gmtMuon.parquet"],
        ["V44nano_GT_MinBias_L1GTgmtTkMuon.parquet", "V44nano_GT_MinBias_L1gmtTkMuon.parquet"],

    ],
    "taus": [
        ["V44nano_GT_MinBias_L1GTnnTau.parquet", "V44nano_GT_MinBias_L1nnPuppiTau.parquet"],
        ["V44nano_GT_VBFHToTauTau_L1GTnnTau.parquet", "V44nano_GT_VBFHToTauTau_L1nnPuppiTau.parquet"],
    ],
    # "met": [
    #     ["V44nano_GT_MinBias_L1GTpuppiMET.parquet", "V44nano_GT_MinBias_L1puppiMET.parquet"],
    #     ["V44nano_GT_TT_L1GTpuppiMET.parquet", "V44nano_GT_TT_L1puppiMET.parquet"],
    # ],
    "photons": [
        ["V44nano_GT_MinBias_L1GTtkPhoton.parquet", "V44nano_GT_MinBias_L1tkPhoton.parquet"],
        ["V44nano_GT_Hgg_L1GTtkPhoton.parquet", "V44nano_GT_Hgg_L1tkPhoton.parquet"]

    ],
}

def get_data(path, file1, file2):
    data1 = ak.from_parquet(f"{path}/{file1}")
    data2 = ak.from_parquet(f"{path}/{file2}")
    return data1, data2

def get_obj_name(filename):
    base = filename.split('.')[0]
    return base.split('_')[-1]

def get_branches(field_lst):
    branch_lst = []
    for field in field_lst:
        branch_lst.append(field.split("_")[-1])
    return branch_lst

def compare_branches(obj1, obj2, branch, data1, data2):
    arr1 = data1[f"{obj1}_{branch}"]
    arr2 = data2[f"{obj2}_{branch}"]
    len_test = ak.sum(ak.num(arr1) != ak.num(arr2))
    entry_idx = ak.where(ak.num(arr1) == ak.num(arr2))
    entry_test = ak.sum(arr1[entry_idx] != arr1[entry_idx])
    
    return len_test, entry_test

def plot_comparison(arr1, arr2, obj1, obj2, branch, datasample, outdir="GT_comparison_plots"):
    import matplotlib.pyplot as plt
    import awkward as ak
    import os

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Flatten arrays for plotting
    flat1 = ak.flatten(arr1, axis=None)
    flat2 = ak.flatten(arr2, axis=None)

    # Convert to NumPy for plotting
    np1 = ak.to_numpy(flat1)
    np2 = ak.to_numpy(flat2)

    # Skip plotting if arrays are empty
    if len(np1) == 0 or len(np2) == 0:
        return

    # Plot
    plt.figure(figsize=(8, 5))

    # Shaded histograms
    plt.hist(np1, bins=100, alpha=0.4, label=obj1, density=True, color='tab:blue')
    plt.hist(np2, bins=100, alpha=0.4, label=obj2, density=True, color='tab:orange')

    # Step outlines (match exact histogram shape)
    plt.hist(np1, bins=100, histtype='step', density=True, color='tab:blue', linewidth=1.5)
    plt.hist(np2, bins=100, histtype='step', density=True, color='tab:orange', linewidth=1.5)

    # Formatting
    plt.title(f"{branch} Comparison ({datasample})")
    plt.xlabel(branch)
    plt.ylabel("Density")
    plt.legend()

    # Save plot
    filename = f"{outdir}/{datasample}_{obj1}_vs_{obj2}_{branch}.png"
    plt.savefig(filename)
    plt.close()



def run(c_path, obj_dict):

    filename = "GT_comparison_results.txt"
    print(f"Writing results to: {os.path.abspath(filename)}")

    with open(filename, "w") as f:
        f.write("Comparison Results:\n")
        f.write("=====================================================================\n")

    for obj, obj_tuples in obj_dict.items():
        with open(filename, "a") as f:
            f.write(f"{obj}: \n")
        for obj_tuple in obj_tuples:
            data1, data2 = get_data(c_path, obj_tuple[0], obj_tuple[1])
            obj1 = get_obj_name(obj_tuple[0])
            obj2 = get_obj_name(obj_tuple[1])
            datasample = obj_tuple[0].split("_")[2]
            with open(filename, "a") as f:
                f.write("-----------------------------------------------------------------------------\n")
                f.write(f"Comparing {obj1} - {obj2} from {datasample}\n")
            branches = set(get_branches(data1.fields)) & set(get_branches(data2.fields))
            for branch in branches: 
                len_test, entry_test = compare_branches(obj1, obj2, branch, data1, data2)
                plot_comparison(data1[f"{obj1}_{branch}"], data2[f"{obj2}_{branch}"],
                obj1, obj2, branch, datasample)
                with open(filename, "a") as f:
                    f.write(f"Branch {branch} has {len_test} mismatches in number of entries and {entry_test} mismatches in values of entries\n")


run(cache_path, obj_dict)




