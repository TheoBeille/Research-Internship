"""Download a few abdominal patients of the Mayo LDCT dataset into ./data/mayo_data."""

from idc_index import IDCClient

client = IDCClient.client()

patients = client.get_patients("ldct_and_projection_data", outputFormat="list")
abdomen_patients = [p for p in patients if p.startswith("L")]
print(f"{len(patients)} patients found, downloading {abdomen_patients[:5]}")

client.download_from_selection(patientId=abdomen_patients[:5],
                               downloadDir="./data/mayo_data")
