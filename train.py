# from data_handler import DataHandler
# from model import create_multi_output_model
# from fruit_shelf_life import PERISHABILITY_LEVELS

# # Initialize parameters
# DATASET_NAME = "moltean/fruits"
# IMG_SIZE = (100, 100)
# BATCH_SIZE = 32

# # Create data handler
# handler = DataHandler(DATASET_NAME, IMG_SIZE, BATCH_SIZE)

# # Download and prepare dataset
# if handler.download_dataset():
#     # Create data generators
#     train_generator = handler.create_data_generator('datasets/Training', augment=True)
#     val_generator = handler.create_data_generator('datasets/Test', augment=False)
    
#     # Get number of classes
#     num_fruits = len(handler.get_class_names())
#     num_perishability_levels = len(PERISHABILITY_LEVELS)
    
#     # Create and train model
#     model = create_multi_output_model(
#         input_shape=(*IMG_SIZE, 3),
#         num_fruits=num_fruits,
#         num_perishability_levels=num_perishability_levels
#     )
    
#     # Train the model
#     history = model.fit(
#         train_generator,
#         validation_data=val_generator,
#         epochs=20,
#         steps_per_epoch=100,
#         validation_steps=50
#     )