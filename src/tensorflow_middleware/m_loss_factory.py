import loss_factories.f_binarycrossentropy_factory as f_binarycrossentropy_factory
import loss_factories.f_binaryfocalcrossentropy_factory as f_binaryfocalcrossentropy_factory
import loss_factories.f_categoricalcrossentropy_factory as f_categoricalcrossentropy_factory
import loss_factories.f_categoricalfocalcrossentropy_factory as f_categoricalfocalcrossentropy_factory
import loss_factories.f_categoricalhinge_factory as f_categoricalhinge_factory
import loss_factories.f_cosinesimilarity_factory as f_cosinesimilarity_factory
import loss_factories.f_ctc_factory as f_ctc_factory
import loss_factories.f_dice_factory as f_dice_factory
import loss_factories.f_hinge_factory as f_hinge_factory
import loss_factories.f_huber_factory as f_huber_factory
import loss_factories.f_kldivergence_factory as f_kldivergence_factory
import loss_factories.f_logcosh_factory as f_logcosh_factory
import loss_factories.f_meanabsoluteerror_factory as f_meanabsoluteerror_factory
import loss_factories.f_meanabsolutepercentageerror_factory as f_meanabsolutepercentageerror_factory
import loss_factories.f_meansquarederror_factory as f_meansquarederror_factory
import loss_factories.f_meansquaredlogarithmerror_factory as f_meansquaredlogarithmerror_factory
import loss_factories.f_poisson_factory as f_poisson_factory
import loss_factories.f_sparsecategoricalcrossentropy_factory as f_sparsecategoricalcrossentropy_factory
import loss_factories.f_squaredhinge_factory as f_squaredhinge_factory
import loss_factories.f_tversky_factory as f_tversky_factory


def new(self, operation: dict) -> None:
    match operation["args"]["class"]:
        case "BinaryCrossentropy":
            f_binarycrossentropy_factory.call(self, operation)
        case "BinaryFocalCrossentropy":
            f_binaryfocalcrossentropy_factory.call(self, operation)
        case "CategoricalCrossentropy":
            f_categoricalcrossentropy_factory.call(self, operation)
        case "CategoricalFocalCrossentropy":
            f_categoricalfocalcrossentropy_factory.call(self, operation)
        case "CategoricalHinge":
            f_categoricalhinge_factory.call(self, operation)
        case "CosineSimilarity":
            f_cosinesimilarity_factory.call(self, operation)
        case "CTC":
            f_ctc_factory.call(self, operation)
        case "Dice":
            f_dice_factory.call(self, operation)
        case "Hinge":
            f_hinge_factory.call(self, operation)
        case "Huber":
            f_huber_factory.call(self, operation)
        case "KLDivergence":
            f_kldivergence_factory.call(self, operation)
        case "LogCosh":
            f_logcosh_factory.call(self, operation)
        case "MeanAbsoluteError":
            f_meanabsoluteerror_factory.call(self, operation)
        case "MeanAbsolutePercentageError":
            f_meanabsolutepercentageerror_factory.call(self, operation)
        case "MeanSquaredError":
            f_meansquarederror_factory.call(self, operation)
        case "MeanSquaredLogarithmError":
            f_meansquaredlogarithmerror_factory.call(self, operation)
        case "Poisson":
            f_poisson_factory.call(self, operation)
        case "SparseCategoricalCrossentropy":
            f_sparsecategoricalcrossentropy_factory.call(self, operation)
        case "SquaredHinge":
            f_squaredhinge_factory.call(self, operation)
        case "Tversky":
            f_tversky_factory.call(self, operation)
        case _:
            raise ValueError(f"Invalid loss class: {operation['args']['class']}")


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = self.new(operation)
