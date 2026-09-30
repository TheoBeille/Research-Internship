from idc_index import IDCClient

client = IDCClient.client()

# Find the exact collection name for LDCT
collections = client.get_collections()


# List patients in the collection
patients = client.get_patients("ldct_and_projection_data", outputFormat="list")
print(f"{len(patients)} patients found")

# Keep patients whose ID starts with L (liver/abdomen)
liver_patients = [p for p in patients if p.startswith("L")]
print(liver_patients[:10])

# Download only 3-5 patients
client.download_from_selection(
    patientId=liver_patients[:5],
    downloadDir="./mayo_data"
)