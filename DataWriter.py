import csv

"""
Class to write data to csv
"""

def write_data_csv(filename : str, data : list[tuple], labels : list = None) -> None:
    listed_data = [list(item) for item in data]
    if labels:
        listed_data = [listed_data[i] + [labels[i]] for i in range(len(data))]
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(listed_data)
