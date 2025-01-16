# run this file once is enough to store annotations
import os
import csv
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from threading import Lock

from preprocess import get_seizure_times, get_raw_data


def generate_annotations(folder_path: str) -> None:
    """
    This function recursively search through all .edf files
    in the folder and its subdirectories to performs 3 tasks:
    1. record patient id and corresponding number of seizures
    in a csv file `patient_summary.csv`.
    2. record all details of the edf files in a 
    `summary.txt` file.
    3. record seizure start and end time stamps
    in `{patient_id}.edf.seizures` file
    for all edf files.

    Args:
        folder_path (str): Path to the folder to search for .edf files.
    """
    patient_seizure_counts = {}
    header = True
    summary_lines = []
    file_lock = Lock()

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        
        # Iterate over all files in the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.edf'):
                    patient_id = os.path.splitext(file)[0]

                    # Submit tasks to the executor
                    futures.append(
                        executor.submit(
                            _process_single_edf, root, patient_id, header, file_lock))

                    if header:
                        header = False

        # Process results as they complete
        for future in as_completed(futures):
            # Retrieve the result for the completed future
            n_seizures, lines, patient_id = future.result()

            # Store seizure count for each patient
            patient_seizure_counts[patient_id] = n_seizures

            # Store the processed lines
            summary_lines.extend(lines)

    # After all tasks are done, write the patient seizure counts
    with open('patient_summary.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Patient_ID", "Seizure_Count"])
        for patient_id, seizure_count in patient_seizure_counts.items():
            writer.writerow([patient_id, seizure_count])

    # Optionally, write summary lines to a separate file if needed
    with open('summary.txt', mode='a') as file:
        for line in summary_lines:
            file.write(line)


def _process_single_edf(root: str, patient_id: str,
                        header:bool,
                        file_lock: Lock) -> Tuple[int, List[str]]:
    raw = get_raw_data(root, patient_id)
    start_time = raw.info['meas_date']
    end_time = start_time + timedelta(seconds=raw.times[-1])
    sampling_rate = raw.info['sfreq']

    summary_lines = []

    if header:  # write header once
        header_lines = []
        header_lines.append(
            f"Data Sampling Rate: {int(sampling_rate)} Hz\n")
        header_lines.append("*************************\n")
        header_lines.append("\nChannels in EDF File:\n")
        header_lines.append("**********************\n")
        for idx, ch_name in enumerate(raw.ch_names):
            header_lines.append(f"Channel {idx + 1}: {ch_name}\n")
        header_lines.append("\n")
        with file_lock:  # avoid conflictions between threads
            with open('summary.txt', mode='w') as file:
                for line in header_lines:
                    file.write(line)
                
    # Get seizures
    seizures = get_seizure_times(
        raw, verbose=False,
        output_path=os.path.join(root, f'{patient_id}.edf.seizures'))
    
    n_seizures = len(seizures)
    
    summary_lines.append(f"\nPatient Id: {patient_id}\n")
    summary_lines.append(f"File Start Time: {start_time.strftime('%H:%M:%S')}\n")
    summary_lines.append(f"File End Time: {end_time.strftime('%H:%M:%S')}\n")
    summary_lines.append(f"Number of Seizures in File: {n_seizures}\n")

    for i, times_dict in enumerate(seizures.values()):
        start_sec = times_dict["start"] / sampling_rate
        end_sec = times_dict["end"] / sampling_rate
        summary_lines.append(f"Seizure {i} Start Time: {start_sec:.1f} seconds\n")
        summary_lines.append(f"Seizure {i} End Time: {end_sec:.1f} seconds\n")
    summary_lines.append("*************************\n")

    return n_seizures, summary_lines, patient_id


if __name__ == '__main__':
    generate_annotations('./data/seizure-data-annotated/')
