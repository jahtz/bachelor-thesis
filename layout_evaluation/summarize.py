import json
from pathlib import Path
import statistics

import click

def print_total(data: list[dict[str, float]]):
    total = {}
    
    for entry in data:
        for key, value in entry.items():
            if key not in total:
                total[key] = 0
            total[key] += value
    
    print(f"{'Total':<11}", *[f"{str(round(value / len(data), 4)):<11}" for key, value in total.items()])
    

def print_stdev(data: list[dict[str, float]]):
    total = {}
        
    for entry in data:
        for key, value in entry.items():
            if key not in total:
                total[key] = []
            total[key].append(value)
    print(f"{'StDev':<11}", *[f"{str(round(statistics.stdev(value), 4)):<11}" for key, value in total.items()])
    


@click.command()
@click.argument(
    "json_files",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    nargs=-1
)
def summarize_cli(json_files: list[Path]):
    header = set()
    result: dict[str, dict[str, float]] = {}
    for fp in sorted(json_files):
        name = fp.name.replace(".json", "")[1:]
        result[name] = {}
        with open(fp, "r") as f:
            data = json.load(f)
        for key, value in data["Total"].items():
            header.add(key.lower())
            result[name][key.lower()] = value
    
    print(f"{'fold':<11}", *[f"{key:<11}" for key in result[name]])
    
    last_name = ""
    temp_metrics = []
    for name, metrics in sorted(result.items(), key=lambda x: x[0]):
        if (name[:3] != last_name) and last_name:
            print_total(temp_metrics)
            print_stdev(temp_metrics)
            temp_metrics = []
            print()
        last_name = name[:3]
        temp_metrics.append(metrics)

        print(f"{name:<11}", *[f"{str(round(value, 4)):<11}" for _, value in metrics.items()])
        
    print_total(temp_metrics)
    print_stdev(temp_metrics)

if __name__ == "__main__":
    summarize_cli()