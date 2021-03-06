# ==================================== #
# TRAIN
# ==================================== #
python train_seglink.py --train_dir weights/table_extract_0530 --dataset_name table_extract --dataset_dir /home/kxu/dataset/SSD-tf/table_extract_test_0528/ --checkpoint_path weights/table_extract_0524_2 --config_file config_table.py

# ==================================== #
# Export inference graph
# ==================================== #
python export_inference_graph.py --checkpoint_path weights/table_extract_0530/model.ckpt-1002 --output_dir inference_graphs/table_extract_0530 --config_file weights/table_extract_0530/train_config.py
