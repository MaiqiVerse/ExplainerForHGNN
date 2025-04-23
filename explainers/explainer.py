class ExplainerCore:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.metrics = config.get("eval_metrics", None)

        self.device = "cpu"
        self.registered_modules_and_params = {}
        self.original_explainer = False

    def init_params(self):
        """Initialize parameters for the explainer."""
        pass

    def init_params_node_level(self):
        """
        Initialize parameters for node-level explanation.
        :return:
        """
        self.edge_mask = None
        self.feature_mask = None
        pass

    def init_params_graph_level(self):
        """
        Initialize parameters for graph-level explanation.
        :return:
        """
        pass

    def extract_neighbors_input(self):
        """
        Extract input of neighbors for the target node.
        :return:
        """
        self.neighbor_input = {"gs": None, "feature": None}
        self.n_hop = self.config.get("n_hop", 3)
        self.used_nodes = []
        self.recovery_dict = {}
        pass

    def explain(self, model, **kwargs):
        """
        Explain the model.
        :param model:
        :param kwargs:
        :return:
        """
        self.model = model
        self.model.eval()

        pass

    def node_level_explain(self, **kwargs):
        """
        Node-level explanation.
        :param kwargs:
        :return:
        """
        self.node_id = kwargs.get("node_id", None)
        if self.node_id is None:
            raise ValueError("Node ID is required for node-level explanation")
        pass

    def graph_level_explain(self):
        """
        Graph-level explanation.
        :return:
        """
        pass

    def visualize(self):
        """
        Visualize the explanation.
        :return:
        """
        pass

    def construct_explanation(self):
        """
        Construct the explanation for metrics and visualization.
        :return:
        """
        pass

    def get_required_fit_params(self):
        """
        Get the required fit parameters, to be used in the optimization process.
        :return:
        """
        pass

    def fit(self):
        """
        Fit the model. If it is a training process, the model will be trained.
        :return:
        """
        pass

    def fit_node_level(self):
        """
        Fit the model for node-level explanation.
        :return:
        """
        pass

    def fit_graph_level(self):
        """
        Fit the model for graph-level explanation.
        :return:
        """
        pass

    def get_loss(self, **kwargs):
        """
        Get the loss for the optimization process.
        :return:
        """
        pass

    def get_loss_node_level(self, **kwargs):
        """
        Get the loss for the optimization process in node-level explanation.
        :return:
        """
        pass

    def get_loss_graph_level(self, **kwargs):
        """
        Get the loss for the optimization process in graph-level explanation.
        :return:
        """
        pass

    def get_input_handle_fn(self):
        """
        Get the input handle function for the model.
        :return:
        """
        pass

    def get_input_handle_fn_node_level(self):
        """
        Get the input handle function for the model in node-level explanation.
        :return:
        """
        self.masked = {"gs": None, "feature": None}
        pass

    def get_input_handle_fn_graph_level(self):
        """
        Get the input handle function for the model in graph-level explanation.
        :return:
        """

        pass

    def forward(self, **kwargs):
        """
        Forward the model.
        :param kwargs:
        :return:
        """
        pass

    def forward_node_level(self, **kwargs):
        """
        Forward the model in node-level explanation.
        :param kwargs:
        :return:
        """
        pass

    def forward_graph_level(self, **kwargs):
        """
        Forward the model in graph-level explanation.
        :param kwargs:
        :return:
        """
        pass

    def build_optimizer(self):
        """
        Build the optimizer for the optimization process.
        :return:
        """
        pass

    def build_scheduler(self, optimizer):
        """
        Build the scheduler for the optimization process.
        :return:
        """
        pass

    @property
    def edge_mask_for_output(self):
        return self.edge_mask

    @property
    def feature_mask_for_output(self):
        return self.feature_mask

    def get_custom_input_handle_fn(self, **kwargs):
        """
        Get the custom input handle function for the model.
        :return:
        """
        pass

    def to(self, device):
        """
        Set the device for the explainer core.
        :param device:
        :return:
        """
        self.device = device
        for module in self.registered_modules_and_params.values():
            module.to(self.device_string)
        return self

    @property
    def device_string(self):
        return "cuda:{}".format(self.device) if self.device != "cpu" else self.device


class Explainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.metrics = config.get("metrics", None)
        self.device = config.get("device", "cpu")
        if self.device is None:
            self.device = "cpu"
        self.registered_modules_and_params = {}

    def explain(self, model):
        self.model = model
        self.model.eval()
        pass

    def node_level_explain(self, **kwargs):
        pass

    def graph_level_explain(self, **kwargs):
        pass

    def construct_explanation(self, result):
        pass

    def evaluate(self):
        pass

    def visualize(self):
        pass

    def get_summary(self):
        pass

    def save_summary(self):
        pass

    def to(self, device):
        """
        Set the device for the explainer.
        :param device:
        :return:
        """
        self.device = device
        for module in self.registered_modules_and_params.values():
            module.to(self.device_string)
        return self

    @property
    def device_string(self):
        return "cuda:{}".format(self.device) if self.device != "cpu" else self.device

    def save_explanation(self):
        pass
