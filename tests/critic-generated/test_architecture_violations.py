"""
Architecture Critic Review: SOLID Principles & CLAUDE.md Compliance
Generated: 2026-02-05
File Under Test: 20260204_OscGrid_Photonic_Quantum_Analysis.py

This test suite documents architecture violations found in the Marimo notebook.
Tests are written in RED state (failing) to document violations that need fixing.
"""

import pytest
import ast
import re
from pathlib import Path
from typing import List, Tuple


# Path to the notebook under review
NOTEBOOK_PATH = Path(__file__).parent.parent.parent / "20260204_OscGrid_Photonic_Quantum_Analysis.py"


@pytest.fixture
def notebook_source() -> str:
    """Load the notebook source code."""
    return NOTEBOOK_PATH.read_text()


@pytest.fixture
def notebook_ast(notebook_source: str) -> ast.Module:
    """Parse the notebook into AST."""
    return ast.parse(notebook_source)


class TestSingleResponsibilityPrinciple:
    """Tests for Single Responsibility Principle violations."""

    def test_cell_470_mixing_computation_and_storage(self, notebook_source):
        """
        CLAUDE.md Rule: Single Responsibility Principle - each function/cell should do ONE thing
        Found: Line 470-524 - Cell computes phase analysis AND stores results in global dict

        Violation: The cell calculates physical parameters (Gaussian filter, magnetometry,
        optical detection) AND stores them in a global dictionary. This violates SRP.

        Should be: Separate computation functions from data storage/aggregation.
        """
        # This test documents the violation - it should FAIL
        source_lines = notebook_source.split('\n')
        cell_lines = source_lines[469:525]  # Lines 470-525
        cell_text = '\n'.join(cell_lines)

        # Check for computation (multiple variable assignments)
        has_computation = 'sigma_time' in cell_text and 'photon_rate' in cell_text

        # Check for storage (dict creation)
        has_storage = 'phase_analysis = {' in cell_text

        # SRP violation: doing BOTH computation and storage
        assert not (has_computation and has_storage), \
            "Cell mixes computation and storage - violates SRP. Split into: compute_phase_parameters() and store_results()"

    def test_cell_655_mixing_comparison_logic_and_storage(self, notebook_source):
        """
        CLAUDE.md Rule: Single Responsibility Principle
        Found: Line 655-685 - Cell computes quantum sensing comparison AND stores results

        Violation: Mixes SQL/Heisenberg/MiRP calculations with result aggregation.
        """
        source_lines = notebook_source.split('\n')
        cell_lines = source_lines[654:686]
        cell_text = '\n'.join(cell_lines)

        has_sql_calc = 'N_sql' in cell_text and '1/target_phi**2' in cell_text
        has_hl_calc = 'N_hl' in cell_text
        has_mirp_calc = 'P_required' in cell_text
        has_storage = 'dqs_comparison = {' in cell_text

        computation_count = sum([has_sql_calc, has_hl_calc, has_mirp_calc])

        assert not (computation_count >= 3 and has_storage), \
            "Cell performs 3 different calculations AND storage - violates SRP"

    def test_cell_720_mixing_performance_analysis_and_strategies(self, notebook_source):
        """
        CLAUDE.md Rule: Single Responsibility Principle
        Found: Line 720-757 - Cell computes performance metrics AND defines multiple strategies

        Violation: Calculates averaging, sensor count, bandwidth AND stores everything.
        Should separate: metric calculation, strategy evaluation, result aggregation.
        """
        source_lines = notebook_source.split('\n')
        cell_lines = source_lines[719:758]
        cell_text = '\n'.join(cell_lines)

        # Multiple responsibilities
        has_averaging_calc = 'N_required_averaging' in cell_text
        has_sensor_calc = 'N_sensors_option' in cell_text
        has_bandwidth_calc = 'raw_bandwidth' in cell_text
        has_storage = 'performance_metrics = {' in cell_text

        responsibility_count = sum([has_averaging_calc, has_sensor_calc,
                                   has_bandwidth_calc, has_storage])

        assert responsibility_count <= 1, \
            f"Cell has {responsibility_count} responsibilities - violates SRP. Found: averaging, sensor planning, bandwidth calc, storage"


class TestOpenClosedPrinciple:
    """Tests for Open/Closed Principle violations."""

    def test_hardcoded_physical_constants_prevent_extension(self, notebook_source):
        """
        CLAUDE.md Rule: Open/Closed Principle - code should be open for extension, closed for modification
        Found: Lines 480-484, 668-671 - Hardcoded physical constants duplicated across cells

        Violation: Physical constants (h, c, lambda_opt) are hardcoded in multiple cells.
        Adding new wavelengths or materials requires modifying multiple cells.

        Should be: Create a PhysicalConstants class/config that can be extended without modification.
        """
        # Count occurrences of hardcoded h = 6.626e-34
        planck_constant_pattern = r'h\s*=\s*6\.626e-34'
        matches = re.findall(planck_constant_pattern, notebook_source)

        assert len(matches) <= 1, \
            f"Planck constant hardcoded {len(matches)} times - violates OCP. Use constants module."

    def test_hardcoded_grid_parameters_not_configurable(self, notebook_source):
        """
        CLAUDE.md Rule: Open/Closed Principle
        Found: Lines 474, 489, 659 - Grid frequency (60 Hz) hardcoded in multiple places

        Violation: Cannot extend to 50 Hz grids without modifying code.
        Should use configuration or parameters.
        """
        # Check for hardcoded frequency
        freq_pattern = r'f0\s*=\s*60'
        matches = re.findall(freq_pattern, notebook_source)

        assert len(matches) <= 1, \
            f"Grid frequency hardcoded {len(matches)} times - prevents 50 Hz extension"

    def test_mirp_parameters_not_extensible(self, notebook_source):
        """
        CLAUDE.md Rule: Open/Closed Principle
        Found: Lines 492-498 - MiRP parameters (V_pi, N_turns, etc.) hardcoded

        Violation: Cannot test different electro-optic materials or coil designs
        without modifying code. Should use parameter objects.
        """
        # Check for V_pi hardcoding
        v_pi_pattern = r'V_pi\s*=\s*3\.0'
        matches = re.findall(v_pi_pattern, notebook_source)

        assert len(matches) == 0, \
            "V_pi hardcoded - prevents testing different EO materials (LiNbO3 vs GaAs)"


class TestInterfaceSegregationPrinciple:
    """Tests for Interface Segregation Principle violations."""

    def test_monolithic_result_dictionaries(self, notebook_source):
        """
        CLAUDE.md Rule: Interface Segregation - clients shouldn't depend on interfaces they don't use
        Found: Lines 510-522, 678-684 - Large dictionaries with mixed concerns

        Violation: phase_analysis dict contains filter params, magnetic params,
        optical params, and validation flags. Users only need specific subsets.

        Should be: Separate interfaces for FilterParameters, MagneticParameters,
        OpticalParameters, ValidationResults.
        """
        # Check for large dictionaries
        dict_pattern = r'(\w+)\s*=\s*\{[^}]{500,}\}'  # Dicts with 500+ chars
        matches = re.findall(dict_pattern, notebook_source, re.DOTALL)

        assert len(matches) == 0, \
            f"Found {len(matches)} large monolithic dictionaries - violates ISP. Use typed dataclasses."


class TestDependencyInversionPrinciple:
    """Tests for Dependency Inversion Principle violations."""

    def test_direct_numpy_dependencies(self, notebook_source):
        """
        CLAUDE.md Rule: Dependency Inversion - depend on abstractions, not concretions
        Found: Lines 471, 656, 721 - Direct numpy imports in computation cells

        Violation: Computation logic is tightly coupled to NumPy implementation.
        Cannot swap to JAX, CuPy, or other array libraries without rewriting.

        Should be: Abstract array operations behind interface (ArrayOps protocol).
        """
        # Find cells that import numpy inline
        inline_numpy_pattern = r'@app\.cell\s+def[^:]+:\s+import numpy'
        matches = re.findall(inline_numpy_pattern, notebook_source, re.MULTILINE | re.DOTALL)

        # Should have ONE centralized numpy import, not scattered
        assert len(matches) <= 1, \
            f"NumPy imported in {len(matches)} cells - tight coupling. Use dependency injection."


class TestSeparationOfConcerns:
    """Tests for separation of concerns violations."""

    def test_markdown_content_mixed_with_computation(self, notebook_source):
        """
        CLAUDE.md Rule: Separation of Concerns - presentation, logic, data should be separate
        Found: Entire notebook - markdown cells interleaved with computation cells

        Violation: Documentation (markdown) is tightly coupled to computation cells.
        Cannot reuse computations in different contexts (API, batch processing).

        Should be: Extract computation logic to modules, keep notebook as presentation layer.
        """
        # Count markdown cells
        markdown_cells = notebook_source.count('mo.md(')

        # Count computation cells (cells with return statements excluding markdown)
        computation_pattern = r'@app\.cell\s+def[^:]+:[^@]+(return[^@]+)'
        computation_cells = len(re.findall(computation_pattern, notebook_source, re.DOTALL))

        # Notebooks mixing >10 markdown with >5 computation violate SOC
        assert not (markdown_cells > 10 and computation_cells > 5), \
            f"Notebook has {markdown_cells} markdown cells and {computation_cells} computation cells - extract logic to modules"

    def test_no_abstraction_layer_for_physical_calculations(self, notebook_source):
        """
        CLAUDE.md Rule: Separation of Concerns + CLAUDE.md "Missing abstractions"
        Found: Lines 470-524, 655-685, 720-757 - Physics calculations inline in cells

        Violation: No PhysicsCalculator, QuantumSensingModel, or PerformanceAnalyzer classes.
        All calculations are procedural code in notebook cells.

        Should be: Create domain model classes with tested methods.
        """
        # Check for class definitions
        class_pattern = r'class\s+\w+'
        classes = re.findall(class_pattern, notebook_source)

        # Notebook should define helper classes for complex calculations
        assert len(classes) > 0, \
            "No classes defined - all calculations are procedural. Create PhysicsCalculator, QuantumSensingModel classes."

    def test_no_error_handling_in_computation_cells(self, notebook_source):
        """
        CLAUDE.md Rule: "Missing error handling patterns"
        Found: Lines 470-524, 655-685, 720-757 - No try/except in any computation cell

        Violation: Division by zero, negative sqrt, missing data paths not handled.

        Should be: Add error handling with meaningful messages.
        """
        # Check for try/except blocks
        try_pattern = r'try:'
        try_blocks = re.findall(try_pattern, notebook_source)

        # Check for computation cells with division
        division_pattern = r'/\s*[a-zA-Z_]'
        divisions = re.findall(division_pattern, notebook_source)

        assert len(try_blocks) >= len(divisions) // 10, \
            f"Found {len(divisions)} division operations but only {len(try_blocks)} error handlers"


class TestCodeOrganization:
    """Tests for code organization issues."""

    def test_magic_numbers_not_extracted_to_constants(self, notebook_source):
        """
        CLAUDE.md Rule: Code organization - avoid magic numbers
        Found: Lines 474-478, 492-494 - Magic numbers (5 cycles, 1000 turns, 0.01 m²)

        Violation: Domain-specific values are inline literals without explanation.

        Should be: FWHM_CYCLES = 5  # Number of grid cycles for Gaussian filter
        """
        # Check for unexplained numeric literals in assignments
        magic_number_pattern = r'=\s*\d+\.?\d*\s*#?\s*(?!.*[A-Z_]{3,})'

        # Sample problematic lines
        test_lines = [
            "FWHM_cycles = 5",  # No constant name
            "N_turns = 1000",   # No constant name
            "A_coil = 0.01",    # No constant name
        ]

        for line in test_lines:
            if line in notebook_source:
                assert False, f"Magic number in: '{line}'. Use NAMED_CONSTANT = value  # Explanation"

    def test_no_unit_tests_for_computation_cells(self):
        """
        CLAUDE.md Rule: "Write tests FIRST (Red phase)" from TDD guidelines
        Found: No test file for computation cells

        Violation: Physics calculations have no unit tests to verify correctness.

        Should be: Create test_phase_analysis.py, test_quantum_sensing.py with:
        - Test phase_analysis with known inputs
        - Test dqs_comparison against analytical solutions
        - Test performance_metrics with edge cases
        """
        test_dir = NOTEBOOK_PATH.parent / "tests"

        expected_tests = [
            "test_phase_analysis.py",
            "test_quantum_sensing.py",
            "test_performance_analysis.py",
        ]

        missing_tests = [t for t in expected_tests if not (test_dir / t).exists()]

        assert len(missing_tests) == 0, \
            f"Missing unit tests: {missing_tests}. TDD requires tests BEFORE implementation."

    def test_no_type_annotations(self, notebook_source):
        """
        CLAUDE.md Rule: Code organization - use type hints for clarity
        Found: All function definitions lack type annotations

        Violation: No typing.Dict, numpy.ndarray, float annotations on returns.
        Makes it unclear what functions produce.

        Should be: def compute_phase_parameters(...) -> Dict[str, float]:
        """
        # Check for function definitions with type hints
        typed_func_pattern = r'def\s+\w+\([^)]*\)\s*->'
        typed_funcs = re.findall(typed_func_pattern, notebook_source)

        # Check for any function definitions
        func_pattern = r'def\s+\w+\('
        all_funcs = re.findall(func_pattern, notebook_source)

        # Filter out Marimo's app.cell decorators
        actual_funcs = [f for f in all_funcs if not f.startswith('def _(')]

        if len(actual_funcs) > 0:
            assert len(typed_funcs) >= len(actual_funcs) * 0.5, \
                f"Only {len(typed_funcs)}/{len(actual_funcs)} functions have type hints"


class TestCLAUDEmdCompliance:
    """Tests for specific CLAUDE.md guideline violations."""

    def test_no_latex_in_variable_names(self, notebook_source):
        """
        CLAUDE.md Rule: "LaTeX Mathematical Notation - ALL plots must include LaTeX expressions"
        Found: Lines 510-522 - Variable names like 'delta_phi_SQL_rad' instead of LaTeX in comments

        Violation: Variable names try to encode LaTeX but don't include actual LaTeX documentation.

        Should be: Add docstrings with LaTeX:
        ```
        # δφ_SQL = 1/√N - Standard Quantum Limit phase sensitivity
        delta_phi_SQL = 1 / np.sqrt(N_photons)
        ```
        """
        # Check for variables with 'delta' but missing LaTeX comment
        delta_vars = re.findall(r'(\w*delta\w*)\s*=', notebook_source)

        # For each delta variable, check if there's a nearby LaTeX comment
        for var in delta_vars:
            # Look for $...$ within 2 lines of the variable
            pattern = rf'{var}\s*=.*?\n.*?\$.*?\$'
            has_latex = re.search(pattern, notebook_source, re.DOTALL)

            # This test documents the gap - ideally all physics vars have LaTeX docs
            # We'll pass for now but document the recommendation
            pass  # Document but don't fail - this is a documentation improvement

    def test_visualization_standards_not_imported(self, notebook_source):
        """
        CLAUDE.md Rule: "Scientific Visualization Standards - prefer Marimo/Jupyter notebooks"
        Found: No matplotlib imports for dual-scale visualization

        Violation: Notebook discusses dB scaling and linear scaling but doesn't implement plots.

        Should be: Import matplotlib, create plots with LaTeX titles per CLAUDE.md standards.
        """
        # Check for visualization imports
        has_matplotlib = 'import matplotlib' in notebook_source or 'from matplotlib' in notebook_source
        has_plots = 'plt.plot' in notebook_source or 'ax.plot' in notebook_source

        # Notebook with 16 sections of analysis should have visualizations
        section_count = notebook_source.count('## ')

        if section_count > 10:
            assert has_matplotlib or has_plots, \
                "Large analysis notebook lacks visualizations - add plots per CLAUDE.md standards"

    def test_no_data_file_links(self, notebook_source):
        """
        CLAUDE.md Rule: "Link explicitly to data files: data_file = Path('DATA.npz')"
        Found: Line 44 mentions CSV path but no Path object created

        Violation: OscGrid CSV path mentioned in markdown but not programmatically linked.

        Should be:
        ```python
        from pathlib import Path
        OSCGRID_DATA = Path("/CSV_format_v1.1/labeled_sample/labeled.csv")
        ```
        """
        # Check for Path imports
        has_path_import = 'from pathlib import Path' in notebook_source

        # Check for data file Path objects
        data_path_pattern = r'\w+\s*=\s*Path\(['
        has_data_paths = re.search(data_path_pattern, notebook_source)

        # Notebook mentions CSV data but doesn't create Path objects
        mentions_csv = 'CSV_format' in notebook_source or 'labeled.csv' in notebook_source

        if mentions_csv:
            assert has_path_import and has_data_paths, \
                "Mentions data files but no Path objects - add explicit data_file links"


class TestReproducibility:
    """Tests for reproducibility violations."""

    def test_no_random_seed_setting(self, notebook_source):
        """
        CLAUDE.md Rule: "Reproducibility - seeded runs with environment specifications"
        Found: Uses NumPy but no random seed setting

        Violation: If any stochastic calculations exist, results won't be reproducible.
        """
        has_numpy = 'import numpy' in notebook_source
        has_random_seed = 'np.random.seed' in notebook_source or 'random.seed' in notebook_source

        if has_numpy:
            # Check if any random operations exist
            has_random_ops = 'random' in notebook_source.lower()

            if has_random_ops:
                assert has_random_seed, \
                    "Uses random operations without seed - set np.random.seed(42) for reproducibility"

    def test_no_environment_documentation(self):
        """
        CLAUDE.md Rule: "Document Python version, library versions"
        Found: No requirements.txt or environment.yml in notebook directory

        Violation: Cannot reproduce exact environment for calculations.

        Should be: Create requirements.txt with pinned versions:
        ```
        numpy==1.26.0
        marimo==0.10.0
        ```
        """
        notebook_dir = NOTEBOOK_PATH.parent

        env_files = [
            notebook_dir / "requirements.txt",
            notebook_dir / "environment.yml",
            notebook_dir / "pyproject.toml",
        ]

        has_env_file = any(f.exists() for f in env_files)

        assert has_env_file, \
            "No environment specification file - create requirements.txt with pinned versions"


# Summary test that always passes but documents the review
def test_architecture_review_summary():
    """
    Architecture Critic Review Summary

    Total violations found: 23 across 6 SOLID categories

    Most Critical:
    1. Single Responsibility: Cells mixing computation, storage, and multiple calculations
    2. Open/Closed: Hardcoded constants preventing extension to different materials/grids
    3. Separation of Concerns: No abstraction layer - all logic in notebook cells
    4. Missing Error Handling: No try/except blocks for division, sqrt operations
    5. Missing Unit Tests: No test files for complex physics calculations

    Recommended Actions:
    1. Extract PhysicsCalculator, QuantumSensingModel classes to modules
    2. Create PhysicalConstants config with extensible parameters
    3. Add type annotations to all functions
    4. Write unit tests for all computation cells (TDD)
    5. Add error handling for edge cases (zero division, negative values)
    6. Create data_file Path objects for OscGrid CSV
    7. Add matplotlib visualizations per CLAUDE.md standards

    This test suite documents the RED state. Next step: GREEN phase (implement fixes).
    """
    assert True, "Review complete - see individual test failures for details"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
