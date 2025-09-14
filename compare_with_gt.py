# compare_with_gt.py
import pandas as pd, os
pred = pd.read_csv("outputs_module4/predictions.csv")
if not os.path.exists("outputs_module2/roi_metadata.csv"):
    print("roi_metadata.csv not found in outputs_module2. Exiting.")
    raise SystemExit
gt = pd.read_csv("outputs_module2/roi_metadata.csv")

# find columns with filename and label in ROI CSV
# common formats: ['filepath','label'] or ['image','label'] etc.
if 'filepath' in gt.columns and 'label' in gt.columns:
    gt['filename'] = gt['filepath'].apply(lambda x: x.split('/')[-1].split('\\')[-1])
elif 'image' in gt.columns and 'class' in gt.columns:
    gt['filename'] = gt['image'].apply(lambda x: x.split('/')[-1].split('\\')[-1])
    gt.rename(columns={'class':'label'}, inplace=True)
elif 'image' in gt.columns and 'label' in gt.columns:
    gt['filename'] = gt['image'].apply(lambda x: x.split('/')[-1].split('\\')[-1])
else:
    # try common fallback: look for a column with "roi" or "file" in its name
    possible = [c for c in gt.columns if 'file' in c.lower() or 'roi' in c.lower() or 'image' in c.lower()]
    if possible:
        gt['filename'] = gt[possible[0]].apply(lambda x: str(x).split('/')[-1].split('\\')[-1])
    else:
        print("Couldn't detect filename column in roi_metadata.csv. Columns:", gt.columns)
        raise SystemExit

# assume label column is 'label' or else attempt to find it
if 'label' not in gt.columns:
    possible_label = [c for c in gt.columns if 'label' in c.lower() or 'class' in c.lower() or 'name' in c.lower()]
    if possible_label:
        gt.rename(columns={possible_label[0]:'label'}, inplace=True)
    else:
        print("Couldn't detect label column in roi_metadata.csv. Columns:", gt.columns)
        raise SystemExit

merged = pred.merge(gt[['filename','label']], on='filename', how='left', suffixes=('','_gt'))
missing_gt = merged['label'].isnull().sum()
print("Merged rows:", len(merged), "| Missing GT rows:", missing_gt)
# where true_label (from ImageFolder) vs label (from CSV) differ:
diffs = merged[merged['true_label'] != merged['label']]
print("Number of rows where folder-true_label != roi_metadata label:", len(diffs))
# compute accuracy using GT CSV label (where available)
valid = merged[merged['label'].notnull()]
acc_vs_csv = (valid['pred_label'] == valid['label']).mean()
print(f"Accuracy comparing predictions to roi_metadata.csv (on {len(valid)} rows): {acc_vs_csv*100:.2f}%")
merged.to_csv("outputs_module4/pred_vs_roi_metadata_merged.csv", index=False)
print("Saved merged CSV: outputs_module4/pred_vs_roi_metadata_merged.csv")
