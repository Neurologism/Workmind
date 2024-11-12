import metric_factories.f_accuracy_factory as f_accuracy_factory
import metric_factories.f_auc_factory as f_auc_factory
import metric_factories.f_binaryaccuracy_factory as f_binaryaccuracy_factory
import metric_factories.f_binarycrossentropy_factory as f_binarycrossentropy_factory
import metric_factories.f_binaryiou_factory as f_binaryiou_factory
import metric_factories.f_categoricalaccuracy_factory as f_categoricalaccuracy_factory
import metric_factories.f_categoricalcrossentropy_factory as f_categoricalcrossentropy_factory
import metric_factories.f_categoricalhinge_factory as f_categoricalhinge_factory
import metric_factories.f_cosinesimilarity_factory as f_cosiinesimilarity_factory
import metric_factories.f_f1score_factory as f_f1score_factory
import metric_factories.f_falsenegatives_factory as f_falsenegatives_factory
import metric_factories.f_falsepositives_factory as f_falsepositives_factory
import metric_factories.f_fbetascore_factory as f_betascore_factory
import metric_factories.f_hinge_factory as f_hinge_factory
import metric_factories.f_iou_factory as f_iou_factory
import metric_factories.f_kldivergence_factory as f_kldivergence_factory
import metric_factories.f_logcosherror_factory as f_logcosh_factory
import metric_factories.f_mean_factory as f_mean_factory
import metric_factories.f_meanabsoluteerror_factory as f_meanabsoluteerror_factory
import metric_factories.f_meanabsolutepercentageerror_factory as f_meanabsolutepercentageerror_factory
import metric_factories.f_meaniou_factory as f_meaniou_factory
import metric_factories.f_meanmetricwrapper_factory as f_meanmetricwrapper_factory
import metric_factories.f_meansquarederror_factory as f_meansquarederror_factory
import metric_factories.f_meansquaredlogarithmicerror_factory as f_meansquaredlogarithmicerror_factory
import metric_factories.f_onehotiou_factory as f_onehotiou_factory
import metric_factories.f_onehotmeaniou_factory as f_onehotmeaniou_factory
import metric_factories.f_poisson_factory as f_poisson_factory
import metric_factories.f_precision_factory as f_precision_factory
import metric_factories.f_precisionatrecall_factory as f_precisionatrecall_factory
import metric_factories.f_r2score_factory as f_r2score_factory
import metric_factories.f_recall_factory as f_recall_factory
import metric_factories.f_recallatprecision_factory as f_recallatprecision_factory
import metric_factories.f_rootmeansquarederror_factory as f_rootmeansquarederror_factory
import metric_factories.f_sensitivityatspecificity_factory as f_sensitivityatspecificity_factory
import metric_factories.f_sparsecategoricalaccuracy_factory as f_sparsecategoricalaccuracy_factory
import metric_factories.f_sparsecategoricalcrossentropy_factory as f_sparsecategoricalcrossentropy_factory
import metric_factories.f_sparsetopkcategoricalaccuracy_factory as f_sparsetopkcategoricalaccuracy_factory
import metric_factories.f_specificityatsensitivity_factory as f_specificityatrecall_factory
import metric_factories.f_squaredhinge_factory as f_squaredhinge_factory
import metric_factories.f_topkcategoricalaccuracy_factory as f_topkcategoricalaccuracy_factory
import metric_factories.f_truenegatives_factory as f_truenegatives_factory
import metric_factories.f_truepositives_factory as f_truepositives_factory

def new(self, operation: dict) -> None:
    match operation["args"]["class"]:
        case "Accuracy":
            f_accuracy_factory.call(self, operation)
        case "AUC":
            f_auc_factory.call(self, operation)
        case "BinaryAccuracy":
            f_binaryaccuracy_factory.call(self, operation)
        case "BinaryCrossentropy":
            f_binarycrossentropy_factory.call(self, operation)
        case "BinaryIOU":
            f_binaryiou_factory.call(self, operation)
        case "CategoricalAccuracy":
            f_categoricalaccuracy_factory.call(self, operation)
        case "CategoricalCrossentropy":
            f_categoricalcrossentropy_factory.call(self, operation)
        case "CategoricalHinge":
            f_categoricalhinge_factory.call(self, operation)
        case "CosineSimilarity":
            f_cosiinesimilarity_factory.call(self, operation)
        case "F1Score":
            f_f1score_factory.call(self, operation)
        case "FalseNegatives":
            f_falsenegatives_factory.call(self, operation)
        case "FalsePositives":
            f_falsepositives_factory.call(self, operation)
        case "FBetaScore":
            f_betascore_factory.call(self, operation)
        case "Hinge":
            f_hinge_factory.call(self, operation)
        case "IOU":
            f_iou_factory.call(self, operation)
        case "KLDivergence":
            f_kldivergence_factory.call(self, operation)
        case "LogCoshError":
            f_logcosh_factory.call(self, operation)
        case "Mean":
            f_mean_factory.call(self, operation)
        case "MeanAbsoluteError":
            f_meanabsoluteerror_factory.call(self, operation)
        case "MeanAbsolutePercentageError":
            f_meanabsolutepercentageerror_factory.call(self, operation)
        case "MeanIOU":
            f_meaniou_factory.call(self, operation)
        case "MeanMetricWrapper":
            f_meanmetricwrapper_factory.call(self, operation)
        case "MeanSquaredError":
            f_meansquarederror_factory.call(self, operation)
        case "MeanSquaredLogarithmicError":
            f_meansquaredlogarithmicerror_factory.call(self, operation)
        case "OneHotIOU":
            f_onehotiou_factory.call(self, operation)
        case "OneHotMeanIOU":
            f_onehotmeaniou_factory.call(self, operation)
        case "Poisson":
            f_poisson_factory.call(self, operation)
        case "Precision":
            f_precision_factory.call(self, operation)
        case "PrecisionAtRecall":
            f_precisionatrecall_factory.call(self, operation)
        case "R2Score":
            f_r2score_factory.call(self, operation)
        case "Recall":
            f_recall_factory.call(self, operation)
        case "RecallAtPrecision":
            f_recallatprecision_factory.call(self, operation)
        case "RootMeanSquaredError":
            f_rootmeansquarederror_factory.call(self, operation)
        case "SensitivityAtSpecificity":
            f_sensitivityatspecificity_factory.call(self, operation)
        case "SparseCategoricalAccuracy":
            f_sparsecategoricalaccuracy_factory.call(self, operation)
        case "SparseCategoricalCrossentropy":
            f_sparsecategoricalcrossentropy_factory.call(self, operation)
        case "SparseTopKCategoricalAccuracy":
            f_sparsetopkcategoricalaccuracy_factory.call(self, operation)
        case "SpecificityAtRecall":
            f_specificityatrecall_factory.call(self, operation)
        case "SquaredHinge":
            f_squaredhinge_factory.call(self, operation)
        case "TopKCategoricalAccuracy":
            f_topkcategoricalaccuracy_factory.call(self, operation)
        case "TrueNegatives":
            f_truenegatives_factory.call(self, operation)
        case "TruePositives":
            f_truepositives_factory.call(self, operation)
        case _:
            raise ValueError(f"Metric {operation['args']['class']} not supported")


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        new(self, operation)