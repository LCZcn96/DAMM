import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def min_max_scale(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data

def preprocess_data(rnaseq_data, methy_data, mirna_data, scnv_data,
                    clinical_data):


    rnaseq_feature_name = rnaseq_data.columns[1:]
    rnaseq_feature_name_new = [name+"_rna" for name in rnaseq_feature_name]
    scaled_data = min_max_scale(rnaseq_data[:, 1:])
    rnaseq_data[rnaseq_feature_name] = scaled_data
    rnaseq_data = rnaseq_data.to_pandas()
    rnaseq_data.columns = ["sample_id"] + rnaseq_feature_name_new

    methy_feature_name = methy_data.columns[1:]
    scaled_data = min_max_scale(methy_data[:, 1:])
    methy_data[methy_feature_name] = scaled_data
    methy_data = methy_data.to_pandas()

    mirna_feature_name = mirna_data.columns[1:]
    scaled_data = min_max_scale(mirna_data[:, 1:])
    mirna_data[mirna_feature_name] = scaled_data
    mirna_data = mirna_data.to_pandas()

    scnv_feature_name = scnv_data.columns[1:]
    # enc = OneHotEncoder(sparse_output=False)
    # cnv_data_encoded = enc.fit_transform(scnv_data[:, 1:])
    # cnv_data_encoded_df = pd.DataFrame(cnv_data_encoded, columns=enc.get_feature_names_out(scnv_feature_name))
    # cnv_data_encoded_df["sample_id"] = scnv_data["sample_id"]
    scnv_data = scnv_data.to_pandas()

    clinical_data = clinical_data.to_pandas()
    clinical_data.dropna(subset = [
        "gender", "years_to_birth", "pathologic_stage", "overall_survival",
        "status"
    ],axis=0, inplace=True)

    # clinical_data.dropna(subset = [
    #      "overall_survival",
    #     "status"
    # ],axis=0, inplace=True)

    clinical_data["gender"] = clinical_data["gender"].apply(
        lambda x: 0 if x == "male" else 1)

    pathologic_stage_map = {
        "stagei": 0,
        "stageii": 1,
        "stageiii": 2,
        "stageiv": 3,
    }

    clinical_data["pathologic_stage"] = clinical_data["pathologic_stage"].map(
        pathologic_stage_map)

    cancer_map = {
        "KIRP": 0,
        "KIRC": 1,
        "KICH": 2,
    }
    clinical_data["cancer_label"] = clinical_data["cancer_label"].map(
        cancer_map)


    merged_data = clinical_data
    merged_data = pd.merge(merged_data, rnaseq_data, on="sample_id")
    merged_data = pd.merge(merged_data, mirna_data, on="sample_id")
    merged_data = pd.merge(merged_data, methy_data, on="sample_id")
    merged_data = pd.merge(merged_data, scnv_data, on="sample_id")

    return rnaseq_feature_name_new, methy_feature_name, mirna_feature_name, scnv_feature_name, merged_data

def load_data():

    clinical_data = pl.read_csv("./data/Clinical.csv")
    methy_data = pl.read_csv("./data/kidney_methylation.csv")
    mirna_data = pl.read_csv("./data/kidney_miRNA.csv")
    rnaseq_data = pl.read_csv("./data/kidney_RNAseq.csv")
    scnv_data = pl.read_csv("./data/kidney_SCNV.csv")
    rnaseq_feature_name,methy_feature_name,mirna_feature_name,scnv_feature_name,merged_data = preprocess_data(rnaseq_data,methy_data,mirna_data,scnv_data,clinical_data)
    # merged_data.set_index("sampleID",inplace=True)

    cancer_data_dict = {
        "merged_data": merged_data,
        "all_feature_name": list(merged_data.columns),
        "rnaseq_feature_name": rnaseq_feature_name,
        "methy_feature_name": methy_feature_name,
        "mirna_feature_name": mirna_feature_name,
        "scnv_feature_name": scnv_feature_name
    }
    print("done")

    return cancer_data_dict




class CancerOmicsDataset(Dataset):
    def __init__(self, rnaseq_data, methy_data, mirna_data, scnv_data, *labels):
        self.rnaseq_data = rnaseq_data
        self.methy_data = methy_data
        self.mirna_data = mirna_data
        self.scnv_data = scnv_data
        self.labels = labels

    def __len__(self):
        return len(self.rnaseq_data)

    def __getitem__(self, idx):
        return self.rnaseq_data[idx],self.methy_data[idx],self.mirna_data[idx],self.scnv_data[idx], [label[idx] for label in self.labels]



def get_cancer_datasets():
    cancer_data_dict = load_data()
    cancer_dataset_dict = {}
    domain_labels = cancer_data_dict["merged_data"]["gender"]
    survival_labels = cancer_data_dict["merged_data"][["overall_survival", "status"]]
    age_labels = cancer_data_dict["merged_data"]["years_to_birth"]
    stage_labels = cancer_data_dict["merged_data"]["pathologic_stage"]
    cancer_type = cancer_data_dict["merged_data"]["cancer_label"]
    rnaseq_data = cancer_data_dict["merged_data"][cancer_data_dict["rnaseq_feature_name"]]
    methy_data = cancer_data_dict["merged_data"][cancer_data_dict["methy_feature_name"]]
    mirna_data = cancer_data_dict["merged_data"][cancer_data_dict["mirna_feature_name"]]
    scnv_data = cancer_data_dict["merged_data"][cancer_data_dict["scnv_feature_name"]]


    # 转换为torch张量
    rnaseq_data = torch.tensor(rnaseq_data.values,
                            dtype=torch.float32,
                            device=device)
    methy_data = torch.tensor(methy_data.values,
                                dtype=torch.float32,
                                device=device)
    mirna_data = torch.tensor(mirna_data.values,
                                dtype=torch.float32,
                                device=device)
    scnv_data = torch.tensor(scnv_data.values,
                                dtype=torch.float32,
                                device=device)
    domain_labels = torch.tensor(domain_labels.values,
                                    dtype=torch.float32,
                                    device=device)
    cancer_type = torch.tensor(cancer_type.values,
                                dtype=torch.int64,
                                device=device)
    survival_labels = torch.tensor(survival_labels.values,
                                    dtype=torch.float32,
                                    device=device)
    age_labels = torch.tensor(age_labels.values,
                                dtype=torch.float32,
                                device=device)
    stage_labels = torch.tensor(stage_labels.values,
                                dtype=torch.float32,
                                device=device)


    dataset = CancerOmicsDataset(
        rnaseq_data, methy_data, mirna_data, scnv_data,
        domain_labels, cancer_type,
        survival_labels, age_labels,
        stage_labels)

    cancer_dataset_dict = {
        "dataset": dataset,
        "rnaseq_feature_name":cancer_data_dict["rnaseq_feature_name"] ,
        "methy_feature_name": cancer_data_dict["methy_feature_name"],
        "mirna_feature_name": cancer_data_dict["mirna_feature_name"],
        "scnv_feature_name": cancer_data_dict["scnv_feature_name"],
    }


    return cancer_dataset_dict
