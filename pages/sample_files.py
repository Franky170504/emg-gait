from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st


@st.cache_data
def create_zip(folder: Path) -> bytes:
    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as zip_file:
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                # Preserve the folder structure inside the ZIP
                archive_path = file_path.relative_to(folder.parent)
                zip_file.write(file_path, archive_path)

    return zip_buffer.getvalue()


folder_path = Path(__file__).parent.parent / "csv"

if folder_path.exists():
    st.download_button(
        label="Download sample files",
        data=create_zip(folder_path),
        file_name="sample_folder.zip",
        mime="application/zip",
    )
else:
    st.error("Sample folder was not found.")

print(folder_path)