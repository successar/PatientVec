# PatientVec

To preprocess data

`python preprocess/process_data.py --main_file [Main csv file] --output_dir preprocess/[Pneumonia|Immunosuppressed] --embedding_file [Path to Pubmed embeddings] --id_field [Name of ID field in main file] --text_field [Name of text field in main file] --label_field [Name of the field containing labels]`


We assume all other columns in the data except the id_field, text_field and label_field contains structured data. The embeddings I am using is http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin .

For example, for Pneumonia dataset,

`python preprocess/process_data.py --main_file [CSV file with pneumonia subset] --output_dir preprocess/Pneumonia --embedding_file [Path to Pubmed embeddings] --id_field [Name of ID field in main file] --text_field [Name of text field in main file] --label_field [Name of the field containing labels]`
