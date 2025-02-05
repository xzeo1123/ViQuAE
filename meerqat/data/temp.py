from datasets import load_dataset, concatenate_datasets

# Định nghĩa danh sách file của dataset
data_files = {
    'humans_with_faces': 'humans_with_faces.jsonl.gz',
    'humans_without_faces': 'humans_without_faces.jsonl.gz',
    'non_humans': 'non_humans.jsonl.gz'
}

# Tải dataset từ Hugging Face
kb = load_dataset('PaulLerner/viquae_wikipedia', data_files=data_files)

# Gán nhãn is_human
kb['humans_with_faces'] = kb['humans_with_faces'].map(lambda item: {'is_human': True})
kb['humans_without_faces'] = kb['humans_without_faces'].map(lambda item: {'is_human': True})
kb['non_humans'] = kb['non_humans'].map(lambda item: {'is_human': False})

# Lưu toàn bộ dữ liệu (100%) vào file
kb_recat = concatenate_datasets([kb['non_humans'], kb['humans_with_faces'], kb['humans_without_faces']])
kb_recat.save_to_disk('data/viquae_wikipedia_recat')

# Lấy 50% dữ liệu của mỗi phần
kb_50percent = {
    'humans_with_faces': kb['humans_with_faces'].train_test_split(test_size=0.5)['test'],
    'humans_without_faces': kb['humans_without_faces'].train_test_split(test_size=0.5)['test'],
    'non_humans': kb['non_humans'].train_test_split(test_size=0.5)['test']
}

# Kết hợp lại và lưu 50%
kb_recat_50percent = concatenate_datasets([kb_50percent['non_humans'], kb_50percent['humans_with_faces'], kb_50percent['humans_without_faces']])
kb_recat_50percent.save_to_disk('data/viquae_wikipedia_recat_50percent')

# Lấy 10% dữ liệu của mỗi phần từ gốc (100%)
kb_10percent = {
    'humans_with_faces': kb['humans_with_faces'].train_test_split(test_size=0.1)['test'],
    'humans_without_faces': kb['humans_without_faces'].train_test_split(test_size=0.1)['test'],
    'non_humans': kb['non_humans'].train_test_split(test_size=0.1)['test']
}

# Kết hợp lại và lưu 10%
kb_recat_10percent = concatenate_datasets([kb_10percent['non_humans'], kb_10percent['humans_with_faces'], kb_10percent['humans_without_faces']])
kb_recat_10percent.save_to_disk('data/viquae_wikipedia_recat_10percent')
