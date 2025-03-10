# run this file once is enough to store annotations
import os
import csv
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from threading import Lock

from utils.preprocess import get_seizure_times, load_raw_data


def generate_annotations(folder_path: str, verbose: bool=False) -> None:
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

    Both summary files will be stored under the folder_path

    Args:
        folder_path (str): Path to the folder to search for .edf files.
    """
    patient_seizure_counts = {}
    header = True
    summary_lines = []
    file_lock = Lock()
    summary_path = os.path.join(folder_path, 'summary.txt')

    with ThreadPoolExecutor() as executor:
        futures = []
        
        # Iterate over all files in the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.edf'):
                    if verbose:
                        print(f'processing {file}')
                    patient_id = os.path.splitext(file)[0]

                    # Submit tasks to the executor
                    futures.append(
                        executor.submit(
                            _process_single_edf, root, patient_id,
                            header, file_lock, summary_path))

                    if header:
                        header = False

        for future in as_completed(futures):
            n_seizures, lines, patient_id = future.result()
            patient_seizure_counts[patient_id] = n_seizures
            summary_lines.extend(lines)

    # write the patient seizure counts
    csv_path = os.path.join(folder_path, 'patient_summary.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Patient_ID", "Seizure_Count"])
        for patient_id, seizure_count in patient_seizure_counts.items():
            writer.writerow([patient_id, seizure_count])

    # write the detailed summary of each file.
    with open(summary_path, mode='a') as file:
        for line in summary_lines:
            file.write(line)

    if verbose:
        print(f'Summary files saved to {folder_path}/summary.txt'
              f' and {folder_path}/patient_summary.csv')
        print(f'Number of patients: {len(patient_seizure_counts)}')


def _process_single_edf(root: str, patient_id: str,
                        header:bool,
                        file_lock: Lock,
                        summary_path: str) -> Tuple[int, List[str]]:
    raw = load_raw_data(root, patient_id)
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
            with open(summary_path, mode='w') as file:
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
    folder_input = input(
        "Enter the folder path in which the edf data files are stored: ")
    generate_annotations(folder_input, verbose=True)
