import os
import pandas as pd

def get_last_two_lines_by_type(filepath, data_type):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        data_lines = []
        is_data_section = False
        for line in lines:
            if line.strip() == data_type:
                is_data_section = True
                continue
            if is_data_section:
                if line.strip():
                    data_lines.append(line.strip())
                else:
                    is_data_section = False
        return data_lines[-2:]

def process_folders(base_dir, prefix):
    results = []
    for folder in os.listdir(base_dir):
        if folder.startswith(prefix):
            folder_path = os.path.join(base_dir, folder)
            results_file = os.path.join(folder_path, 'result.txt')
            if os.path.exists(results_file):
                last_two_ood_lines = get_last_two_lines_by_type(results_file, 'OOD')
                if len(last_two_ood_lines) < 2:
                    continue
                
                folder_name = folder[len(prefix):]
                folder_name = folder_name[:folder_name.rfind('-')]
                
                method_name = folder_name[1:]
                
                mean_values = list(map(float, last_two_ood_lines[0].strip('[]').split(',')))
                std_values = list(map(float, last_two_ood_lines[1].strip('[]').split(',')))

                results.append({
                    'Method': method_name,
                    'Type': 'Mean',
                    'MAE': mean_values[0],
                    'RSE': mean_values[1],
                    'PCC': mean_values[2],
                    'KCC': mean_values[3]
                })
                results.append({
                    'Method': method_name,
                    'Type': 'Std',
                    'MAE': std_values[0],
                    'RSE': std_values[1],
                    'PCC': std_values[2],
                    'KCC': std_values[3]
                })
    return results

def main():
    base_dir = './runs'
    all_mma_results = process_folders(base_dir, 'all')
    abla_feature_mma_results = process_folders(base_dir, 'abla_feature-mma')
    abla_data_mma_results = process_folders(base_dir, 'abla_data-mma')

    with pd.ExcelWriter('results.xlsx') as writer:
        if all_mma_results:
            df_all_mma = pd.DataFrame(all_mma_results)
            df_all_mma.to_excel(writer, sheet_name='all-mma', index=False)
        else:
            pd.DataFrame().to_excel(writer, sheet_name='all-mma', index=False)

        if abla_feature_mma_results:
            df_abla_feature_mma = pd.DataFrame(abla_feature_mma_results)
            df_abla_feature_mma.to_excel(writer, sheet_name='abla_feature-mma', index=False)
        else:
            pd.DataFrame().to_excel(writer, sheet_name='abla_feature-mma', index=False)

        if abla_data_mma_results:
            df_abla_data_mma = pd.DataFrame(abla_data_mma_results)
            df_abla_data_mma.to_excel(writer, sheet_name='abla_data-mma', index=False)
        else:
            pd.DataFrame().to_excel(writer, sheet_name='abla_data-mma', index=False)

if __name__ == "__main__":
    main()