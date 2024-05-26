"""
Dataloader class for a "Regression with a Flood Prediction Dataset" 
https://www.kaggle.com/competitions/playground-series-s4e5
"""

import pandas as pd


class DataLoader:
    def __init__(self, train_path, test_path, sample_submission_path=None):
        self.train_path = train_path
        self.test_path = test_path
        self.sample_submission_path = sample_submission_path

    def _load_csv(self, path):
        return pd.read_csv(path, index_col="id")

    def get_train_set(self):
        train_set = self._load_csv(self.train_path)
        self.base_features = train_set.columns[:-1]
        target_column = train_set.columns[-1]
        X = train_set[self.base_features]
        y = train_set[target_column]
        X = self.add_features(X)

        return X, y

    def get_test_set(self):
        test_set = self._load_csv(self.test_path)
        self.base_features = test_set.columns
        X = test_set[self.base_features]
        X = self.add_features(X)
        return X

    def add_features(self, df):
        df["total"] = df[self.base_features].sum(axis=1)
        df["mean"] = df[self.base_features].mean(axis=1)
        df["max"] = df[self.base_features].max(axis=1)
        df["min"] = df[self.base_features].min(axis=1)
        df["median"] = df[self.base_features].median(axis=1)
        df["ptp"] = df[self.base_features].values.ptp(axis=1)
        df["q25"] = df[self.base_features].quantile(0.25, axis=1)
        df["q75"] = df[self.base_features].quantile(0.75, axis=1)
        df["ClimateImpact"] = df["MonsoonIntensity"] + df["ClimateChange"]
        df["AnthropogenicPressure"] = (
            df["Deforestation"]
            + df["Urbanization"]
            + df["AgriculturalPractices"]
            + df["Encroachments"]
        )
        df["InfrastructureQuality"] = (
            df["DamsQuality"]
            + df["DrainageSystems"]
            + df["DeterioratingInfrastructure"]
        )
        df["CoastalVulnerabilityTotal"] = df["CoastalVulnerability"] + df["Landslides"]
        df["PreventiveMeasuresEfficiency"] = (
            df["RiverManagement"]
            + df["IneffectiveDisasterPreparedness"]
            + df["InadequatePlanning"]
        )
        df["EcosystemImpact"] = df["WetlandLoss"] + df["Watersheds"]
        df["SocioPoliticalContext"] = df["PopulationScore"] * df["PoliticalFactors"]
        df["FloodVulnerabilityIndex"] = (
            df["AnthropogenicPressure"]
            + df["InfrastructureQuality"]
            + df["CoastalVulnerabilityTotal"]
            + df["PreventiveMeasuresEfficiency"]
        ) / 4

        df["AgriculturalEncroachmentImpact"] = (
            df["AgriculturalPractices"] * df["Encroachments"]
        )

        df["DamDrainageInteraction"] = df["DamsQuality"] * df["DrainageSystems"]

        df["LandslideSiltationInteraction"] = df["Landslides"] * df["Siltation"]

        df["PoliticalPreparednessInteraction"] = (
            df["PoliticalFactors"] * df["IneffectiveDisasterPreparedness"]
        )

        df["TopographyDrainageSiltation"] = df["TopographyDrainage"] + df["Siltation"]

        df["ClimateAnthropogenicInteraction"] = (
            df["ClimateImpact"] * df["AnthropogenicPressure"]
        )

        df["InfrastructurePreventionInteraction"] = (
            df["InfrastructureQuality"] * df["PreventiveMeasuresEfficiency"]
        )

        df["CoastalEcosystemInteraction"] = (
            df["CoastalVulnerabilityTotal"] * df["EcosystemImpact"]
        )

        return df
