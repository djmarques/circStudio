"""
Tab 0b - Batch overview (grouped metrics & plots).

This module computes summary circadian metrics for every subject in a loaded
batch, displays the results in a table and boxplot, and allows the user to
select the subject used by the individual-level analysis tabs.
"""
# Libraries for data processing and plotting
import pandas as pd
import plotly.graph_objects as go

# Shiny imports for UI and server functionality
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

# Library imports used to locate and import the circStudio package
from pathlib import Path
from import_paths import _add_import_path

# Add the parent directory of this file to the import path to ensure that
# the circStudio package can be imported correctly
app_dir = Path(__file__).resolve().parent.parent
_add_import_path(app_dir)

# Import circStudio modules after adding the import path
from circstudio.analysis import IS, IV, l5, m10, ra  # noqa: E402

# Import a utility function for creating empty figures from the common module
from modules._common import empty_fig  # noqa: E402

# Define the actigraphy metrics that will be computed and displayed in the 
# batch overview tab
_METRICS = ["IS", "IV", "L5", "M10", "RA"]

# ---------------------------------------------------------------------
# Actigraphy metric calculation
# ---------------------------------------------------------------------
def _compute_metrics(series) -> dict:
    """Compute five actigraphy metrics for one subject's activity series.

    Parameters
    ----------
    series
        Time-indexed activity measurements for a single subject.

    Returns
    -------
    dict
        IS, IV, L5, M10, and RA values. Metrics that cannot be calculated
        are represented by ``NaN`` so that the remaining results can still
        be displayed.
    """
    # Store all calculated metrics for the current subject
    metrics = {}

    # calculate IS and IV metrics, handling any exceptions that may arise
    for name, fn in (("IS", IS), ("IV", IV)):
        try:
            # Convert the result to a float and store it in the metrics dictionary
            metrics[name] = float(fn(series))
        
        # Represent failed calculations as missing without stopping the batch
        except Exception:
            metrics[name] = float("nan")
    
    # Calculate L5 and M10 metrics, handling any exceptions that may arise
    for name, fn in (("L5", l5), ("M10", m10)):
        try:
            # Run the selected metric on the activity series
            res = fn(series)

            # Convert the result to a float and store it in the metrics dictionary
            metrics[name] = float(res[1] if isinstance(res, tuple) else res)
        
        # Represent failed calculations as missing without stopping the batch
        except Exception:
            metrics[name] = float("nan")

    # Calculate RA metric, handling any exceptions that may arise
    try:
        # Run the selected metric on the activity series
        metrics["RA"] = float(ra(series))
    
    # Represent failed calculations as missing without stopping the batch
    except Exception:
        metrics["RA"] = float("nan")
    
    # Return the dictionary containing all computed metrics for the current subject
    return metrics

# ---------------------------------------------------------------------
# Batch overview user interface
# ---------------------------------------------------------------------
# Register this function as the UI component of a reusable Shiny module
@module.ui
def batch_overview_ui():
    """Create the controls and outputs for the batch overview tab."""
    # Arrange the controls in a sidebar and the results in the main panel
    return ui.layout_sidebar(
        ui.sidebar(
            # Let the user start metric calculations for the batch
            ui.input_action_button(
                "run_metrics",
                "Run metrics on all subjects",
                class_="btn-primary btn-sm",
            ),
            ui.hr(),

            # Select which actigraphy metric is displayed in the boxplot
            ui.input_select(
                "box_metric",
                "Boxplot metric",
                choices=_METRICS,
                selected="IS"
            ),

            # Insert the subject selector generated from the loaded batch
            ui.output_ui("subject_picker"),
            width=320,
        ),
        # Explain the purpose of the tab and of the active subject selector
        ui.div(
            ui.p(
                "This tab summarises interdaily stability, intradaily variability, "
                "and other circadian metrics computed across all loaded subjects. "
                "Use the subject selector at the bottom of the sidebar to set the "
                "active recording used by the individual-level analysis tabs."
            ),
            class_="text-muted small mb-3",
        ),
        # Display the calculated metrics as a dataframe
        ui.h4("Grouped metrics"),
        ui.output_data_frame("metrics_table"),
        ui.hr(),
        # Display the selected metric grouped by the first batch factor
        ui.h4("Metric distribution by factor"),
        output_widget("metrics_box"),
    )

# ---------------------------------------------------------------------
# Batch overview server logic
# ---------------------------------------------------------------------
# Register this function as the server logic for the Shiny module
@module.server
def batch_overview_server(input, output, session, rv_batch, rv_active_subject):
    """Server logic for the batch overview tab"""
    # Cache the metrics DataFrame so it can be shared by the table and plot
    _results = reactive.Value(None)

    # Recalculate metrics only when the user clicks the run button
    @reactive.effect
    @reactive.event(input.run_metrics)
    def _run():
        """Retrieve the currently loaded batch"""
        batch = rv_batch()

        # Stop if there are no subject recordings to analyze
        if batch is None or len(batch) == 0:
            ui.notification_show("No batch loaded.", type="warning")
            return None
        
        # Collect one results row for each subject
        rows = []

        # Process each subject independently so its metadata remains aligned
        for entry in batch.entries:
            # Start the row with the subject identifier
            row = {"subject_id": entry.subject_id}

            # Add the subject's level for every factor defined in the batch
            for factor_index, factor_name in enumerate(batch.factor_names):
                # Use the corresponding level when it is available
                if factor_index < len(entry.factor_levels):
                    factor_level = entry.factor_levels[factor_index]

                # Use an empty label when the subject lacks this factor level
                else:
                    factor_level = ""

                # Store the factor level under its factor name
                row[factor_name] = factor_level

            # Calculate and add the five metrics for this activity recording
            row.update(_compute_metrics(entry.raw.activity))
            rows.append(row)
        
        # Convert the subject-level records into a DataFrame
        results_df = pd.DataFrame(rows)

        # Store the results DataFrame in the reactive value so it can be accessed 
        # by the table and plot
        _results.set(results_df)

        # Inform the user how many subjects were successfully processed
        ui.notification_show(
            f"Computed metrics for {len(rows)} subject(s).",
            type="message"
        )

    # Render the cached subject-level metrics as a DataFrame in the UI
    @render.data_frame
    def metrics_table():
        """Display the calculated metrics for all subjects"""
        df = _results.get()

        # Show an instruction until the user requests the calculations
        if df is None:
            return pd.DataFrame(
                {"info": ["Click 'Run metrics on all subjects'."]}
            )
        
        # Copy the results so display formatting does not alter the cache
        show = df.copy()

        # Round metric values to four decimal places for readability
        for metric in _METRICS:
            if metric in show:
                show[metric] = show[metric].round(4)
        
        # Return an interactive grid that fills the available width
        return render.DataGrid(show, width="100%")

    # Render the selected metric as a Plotly boxplot
    @render_widget
    def metrics_box():
        """Plot the distribution of one metric across subjects and groups"""
        df = _results.get()

        # Display a placeholder until the metrics have been computed
        if df is None:
            return empty_fig(
                "Run metrics to see the distribution."
            )

        # Read the metric selected in the sidebar
        metric = input.box_metric()

        # Guard against a metric being absent from the results
        if metric not in df:
            return empty_fig("Metric not available.")

        # Retrieve the batch metadata used to identify factors
        batch = rv_batch()
        
        # Use the first factor in the batch to group the boxplot, if available
        if batch:
            factor = batch.factor_names[0]
        else:
            factor = None
        
        # Create an empty Plotly figure
        fig = go.Figure()

        # Draw a separate boxplot for each level of the factor
        if factor and factor in df:
            for factor_level, group_df in df.groupby(factor):
                fig.add_trace(
                    go.Box(
                        y=group_df[metric],
                        name=str(factor_level),
                        boxpoints="all"
                    )
                )
            
            # Label the horizontal axis with the factor name
            fig.update_layout(xaxis_title=factor)
        
        # Draw a single boxplot when the batch has no factor for grouping
        else:
            fig.add_trace(
                go.Box(
                    y=df[metric],
                    name="all",
                    boxpoints="all"
                )
            )
        
        if factor:
            plot_title = f"{metric} by {factor}"
        else:
            plot_title = metric
        
        # Include the factor in the title when one is being used
        fig.update_layout(
            yaxis_title=metric,
            title=plot_title,
            margin={
                "l": 40,
                "r": 20,
                "t": 40,
                "b": 30
            },
        )
        
        # Return the completed interactive figure
        return fig


    # -----------------------------------------------------------------
    # Active subject selection
    # -----------------------------------------------------------------
    # Render the selector dynamically because its choices depend on the batch
    @render.ui
    def subject_picker():
        """Create the subject selector from the loaded batch identifiers"""
        # Retrieve the currently loaded batch to populate the selector
        batch = rv_batch()

        # Show a placeholder when no subject recordings are available
        if batch is None or len(batch) == 0:
            return ui.p(
                "No batch loaded."
            )
        
        # Populate the selector with all subject identifiers in the batch
        return ui.input_select(
            "active_subject",
            "Active subject (used by analysis tabs)",
            choices=batch.subject_ids(),
        )

    # Keep the shared active subject value synchronized with the selector
    @reactive.effect
    def _sync_active_subject():
        """Share the selected subject with the individual analysis tabs"""
        try:
            # Read the selector value after the dynamic UI has been created
            val = input.active_subject()
        
        # Exit quietly if the selector is not yet available (e.g., no batch loaded)
        except Exception:
            return None
        
        # Update the shared value only when a valid subject was selected
        if val:
            rv_active_subject.set(val)
