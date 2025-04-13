from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, udf, row_number
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
import os
import shutil
import re


def classify_vehicle_type(plate):
    if not plate:
        return "nezinomas"

    plate = plate.strip().upper()

    if re.fullmatch(r"T\d{5}", plate):
        return "taksi"

    if re.fullmatch(r"E[A-Z]\d{4}", plate):
        return "elektromobilis"

    if re.fullmatch(r"H\d{4,5}", plate):
        return "istorinis"

    if re.fullmatch(r"\d{6}", plate):
        return "diplomatinis"

    if re.fullmatch(r"[A-Z]{3}\d{3}", plate):
        return "lengvasis"

    if re.fullmatch(r"[A-Z]{2}\d{3}", plate):
        return "priekaba"

    if re.fullmatch(r"\d{3}[A-Z]{2}", plate):
        return "motociklas"

    if re.fullmatch(r"\d{2}[A-Z]{3}", plate):
        return "mopedas"

    if re.fullmatch(r"\d{2}[A-Z]{2}", plate):
        return "keturratis"

    if len(plate) <= 6 and any(char.isdigit() for char in plate):
        return "personalizuotas"

    return "nezinomas"


vehicle_type_udf = udf(classify_vehicle_type, StringType())


def main():
    spark = SparkSession.builder.appName("VDA_NumberPlatePipeline").getOrCreate()

    df1 = spark.read.csv("df1.csv", header=True).withColumn("failo_vardas", lit("df1"))
    df2 = spark.read.csv("df2.csv", header=True).withColumn("failo_vardas", lit("df2"))
    df3 = spark.read.csv("df3.csv", header=True).withColumn("failo_vardas", lit("df3"))

    combined_df = df1.union(df2).union(df3)
    _save_single_csv(combined_df, "combined.csv", temp_dir="temp_combined")
    print("combined.csv created")

    combined_df_unique = combined_df.dropDuplicates(["numeris"])
    _save_single_csv(combined_df_unique, "combined_unique.csv", temp_dir="temp_unique")
    print("Combined_unique.csv created (duplicates removed)")

    df_classified = combined_df_unique.withColumn(
        "tipas", vehicle_type_udf(combined_df_unique["numeris"])
    )

    window_spec = Window.orderBy("numeris")
    df_classified = df_classified.withColumn("eil_nr", row_number().over(window_spec))

    _save_single_csv(
        df_classified, "combined_classified.csv", temp_dir="temp_classified"
    )
    print("Combined_classified.csv created with vehicle type and row numbers")

    spark.stop()


def _save_single_csv(df, final_filename, temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    df.coalesce(1).write.csv(temp_dir, header=True, mode="overwrite")

    for file in os.listdir(temp_dir):
        if file.startswith("part-") and file.endswith(".csv"):
            if os.path.exists(final_filename):
                os.remove(final_filename)
            shutil.move(os.path.join(temp_dir, file), final_filename)
            break

    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
