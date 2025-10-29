from cnnClassifier.constants import *
import os
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                 PrepareBaseModelConfig,
                                                 TrainingConfig,
                                                 EvaluationConfig)
from pathlib import Path  # تم الإبقاء على استيراد Path
from tensorflow import keras # تم إضافة استيراد Keras هنا


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    # دالة مساعدة لإنشاء النموذج الكامل
    # (تم نقلها هنا للمحافظة على بنية الكود)
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False

        flatten_in = keras.layers.Flatten()(model.output)
        prediction = keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        return full_model


    # 1. دالة تهيئة النموذج الأساسي (Prepare Base Model)
    def prepare_base_model(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params

        self.model = keras.applications.vgg16.VGG16(
            input_shape=config.IMAGE_SIZE,
            weights="imagenet",
            include_top=config.INCLUDE_TOP
        )

        # تجميد جميع طبقات VGG16
        for layer in self.model.layers:
            layer.trainable = False

        # **********************************************
        # *********** إلغاء تجميد آخر 4 طبقات (Fine-tuning) ***********
        # **********************************************
        
        for layer in self.model.layers[-4:]:
            layer.trainable = True
            
        # بناء النموذج الكامل
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=params.CLASSES,
            freeze_all=False,
            learning_rate=params.LEARNING_RATE
        )

        # حفظ النموذج المعدل
        self.save_model(path=config.base_model_path, model=self.full_model)

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            image_size=config.IMAGE_SIZE,
            learning_rate=config.LEARNING_RATE,
            include_top=config.INCLUDE_TOP,
            weights=config.WEIGHTS,
            classes=config.CLASSES
        )
        
    def save_model(self, path: Path, model: keras.Model):
        model.save(path)


    # 2. دالة إعداد التدريب (Get Training Config)
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config
    

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/kidney-ct-scan-image",
            mlflow_uri="https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config