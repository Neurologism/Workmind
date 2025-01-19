from .m_dependencies import *
from .m_layer_factory import call as layer_factory_call
from .m_model_factory import call as model_factory_call
from .m_dataset_factory import call as dataset_factory_call
from .m_initializer_factory import call as initializer_factory_call
from .m_regularizer_factory import call as regularizer_factory_call
from .m_constraint_factory import call as constraint_factory_call
from .c_callbacks import DatabaseLogger
from .m_visualizer_factory import call as visualizer_factory_call


class WhitemindProject:
    def __init__(
        self, json_data: dict | None = None, log_function=None, task_id="test"
    ) -> None:
        self.database_logger = DatabaseLogger(log_function, self)
        self.task_id = task_id

        self.layers = []
        self.model_operations = []
        self.datasets = []
        self.visualizers = []

        self.execution_head = None

        # build the project objects here out of helper classes

    def __call__(self):
        while self.execution_head is not None:
            self.execution_head()

        # execute the project here with call functions of the helper classes
