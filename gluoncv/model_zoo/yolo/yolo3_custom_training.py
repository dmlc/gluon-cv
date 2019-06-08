class YOLOV3CustomTraining:
    """YOLO V3 Custom Training API
    Parameters
    ---------
    classes: iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    train_data: RecordFileDetection
        Instance of RecordFileDetection with training data
    val_data: RecordFileDetection
        Instance of RecordFileDetection with training data
    num_samples:
        Set the number of traning images used
    randomize:
        Use RandomTransformDataLoader insted of DataLoader
    """
    def __init__(self, classes, train_data, val_data, num_samples=-1, randomize=True):
        pass

    @property
    def best_params(self):
        """Get the best params filename"""

    def set_params_object(self, params):
        """Set parameters object. This is for passing args directly"""        

    def set_params(self, data_shape=416, batch_size=64, num_workers=4,
                   epochs=200, lr=0.001, lr_mode='step', lr_decay=0.1, 
                   lr_decay_period=0, lr_decay_epoch=[160,180], 
                   warmup_lr=0.0, warmpu_epochs=0,
                   momentum=0.9, wd=0.0005, seed=233):
        """Set training parameters"""        
    
    def set_resume(self, resume=None, start_epoch=0):
        """Set resume training"""        

    def set_logging_options(self, stdout=True, log_prefix='', log_interval=100):
        """Set logging options. If stdout == False, save to file"""        

    def set_save_options(self, save_prefix='', save_interval=10):
        """Set save options"""        

    def set_gpu_context(self, gpus=[0], syncbn=False):
        """Set context to GPUS and if should use synchonized batch normalization"""        

    def set_cpu_context(self):
        """Set context to CPU"""        

    def disable_random_shape(self):
        """Use fixed size(data-shape) throughout the training, which will be faster 
        and require less memory. However, final model will be slightly worse."""        

    def disable_wd(self):
        """Disable weigth decay"""

    def enable_mixup(self, no_mixup_epochs=20):
        """Enable mixup and disable for the last N epochs"""        

    def enable_label_smoothing(self):
        """Enable label smoothing"""        

    def set_train_custom_transform(self, custom_transform, randomize=True):
        """Pass a custom transform class for training. See YOLO3DefaultTrainTransform to create your own"""        
    
    def set_val_custom_transform(self, custom_transform, randomize=True):
        """Pass a custom transform class for validation. See YOLO3DefaultValTransform to create your own"""

    def set_custom_eval_metric(self, eval_metric):
        """Pass an instance of mx.metric.EvalMetric to perform custom metrics"""

    def set_net_params(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        """Set some net params"""

    def start_training(self):
        """Start training"""
        pass    

    def export(filename):
        """Export the network and best params with given filename"""

    def _get_dataloader(self):
        pass

    def _save_params(self, best_map, current_map, epoch, save_interval, prefix):
        pass
    
    def _validate(self):
        pass

    def _train(self):
        pass

