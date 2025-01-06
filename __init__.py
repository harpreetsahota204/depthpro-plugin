import os

from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from depth_pro_inference import (
        run_depth_prediction,
    )

def _handle_calling(
        uri, 
        sample_collection, 
        depth_field,
        depth_type,
        delegate=False
        ):
    ctx = dict(dataset=sample_collection)

    params = dict(
        depth_field=depth_field,
        depth_type=depth_type,
        delegate=delegate
        )
    return foo.execute_operator(uri, ctx, params=params)

class DepthProEstimator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            # The operator's URI: f"{plugin_name}/{name}"
            name="depth_pro_estimator",  # required

            # The display name of the operator
            label="Depth estimation via Depth Pro",  # required

            # A description for the operator
            description="Perfom zero-shot metric monocular depth estimation using the Apple Depth Pro model",

            icon="/assets/depth-perception.svg",

            )

    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()

        depth_types = types.RadioGroup(label="Which depth map representation do you want to use?",)

        depth_types.add_choice(
            "regular", 
            label="Regular depth",
            description="Direct physical distance measurement in meters. Preferred for autonomous driving, precise measurements, and safety-critical applications. Better for accurate obstacle detection and motion planning."
        )

        depth_types.add_choice(
            "inverse", 
            label="Inverse depth",
            description="Reciprocal of depth (1/distance). Better for visualizing near-field details, indoor environments, and SLAM applications. Provides more detail in close ranges where depth changes are significant."
        )
        
        inputs.enum(
            "depth_type",
            values=depth_types.values(),
            view=depth_types,
            required=True
        )

        inputs.str(
            "depth_field",            
            required=True,
            description="Name of the field to store the depth map."
            )
        
        inputs.bool(
            "delegate",
            default=False,
            required=True,
            label="Delegate execution?",
            description=("If you choose to delegate this operation you must first have a delegated service running. "
            "You can launch a delegated service by running `fiftyone delegated launch` in your terminal"),
            view=types.CheckboxView(),
        )

        inputs.view_target(ctx)

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        """Implement this method if you want to programmatically *force*
        this operation to be delegated or executed immediately.

        Returns:
            whether the operation should be delegated (True), run
            immediately (False), or None to defer to
            `resolve_execution_options()` to specify the available options
        """
        return ctx.params.get("delegate", False)


    def execute(self, ctx):
        """Executes the actual operation based on the hydrated `ctx`.
        All operators must implement this method.

        This method can optionally be implemented as `async`.

        Returns:
            an optional dict of results values
        """
        view = ctx.target_view()
        depth_field = ctx.params.get("depth_field")
        depth_type = ctx.params.get("depth_type")
        
        run_depth_prediction(
            dataset=view,
            depth_field=depth_field,
            depth_type=depth_type,
            )
        
        ctx.ops.reload_dataset()

    def __call__(
            self, 
            sample_collection, 
            depth_field, 
            depth_type,
            delegate=False
            ):
        return _handle_calling(
            self.uri,
            sample_collection,
            depth_field,
            depth_type,
            delegate=delegate
            )

def register(p):
    """Always implement this method and register() each operator that your
    plugin defines.
    """
    p.register(DepthProEstimator)