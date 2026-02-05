# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=1.24.0",
#     "matplotlib>=3.7.0",
#     "networkx>=3.0",
#     "folium>=0.14.0",
#     "pandas>=2.0.0",
# ]
# ///

import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # BARQNET: Boston-Area Quantum Network for IoCQT Applications

    ## Two-Node Boston–MARQI Hybrid Quantum Network Architecture

    **Document Version**: 2026-02-05
    **Project**: ROAD-IoCQT STC - Internet of Classical and Quantum Technologies

    ---

    This notebook details the proposed two-node quantum network deployment connecting:

    1. **Node 1 (Boston/MIT/CQN)**: Silicon-vacancy (SiV) quantum memories, NV-cavity RF/magnetometry, MiRP receivers, ZALM entangled-photon sources
    2. **Node 2 (MARQI/UMD)**: Trapped-ion nodes, neutral-atom nodes, atomic clocks, OPM/Rydberg RF sensors

    The architecture enables large-scale Internet of Classical and Quantum Technologies (IoCQT) applications and distributed Classical-Quantum Machine Learning (CQML).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Network Scale and Sufficiency

    ### BARQNET: 50 km Boston-Area Fiber Testbed

    The Boston-Area Quantum Network (BARQNET) provides a **50 km fiber testbed** with entanglement-compatible transmission characteristics:

    | Parameter | Specification | Notes |
    |-----------|--------------|-------|
    | Total fiber length | ~50 km | Deployed SMF-28 fiber |
    | Number of nodes | 24 | See network topology below |
    | Transmission loss | < 0.2 dB/km | Standard telecom fiber |
    | Entanglement fidelity target | > 0.9 | After error correction |
    | Memory coherence ($T_2$) | > 1 ms (SiV), > 1 s (trapped ions) | Platform dependent |

    ### Why 50 km is Sufficient for Initial Deployment

    1. **Metropolitan-scale coverage**: Spans MIT, Harvard, Lincoln Lab, and partner institutions
    2. **Realistic loss regime**: Total loss ~10 dB matches near-term repeater capabilities
    3. **Scalability testbed**: Validates protocols before continental extension
    4. **Multi-platform integration**: Connects heterogeneous quantum systems
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass, field
    from typing import Optional
    import networkx as nx
    return dataclass, field, np, nx, plt, Optional


@app.cell
def _(dataclass, field):
    # Node definitions from quantum-network-code repository
    # Source: https://github.com/dirkenglund/quantum-network-code

    @dataclass
    class QuantumNode:
        """Quantum network node specification."""
        uid: str
        name: str
        position: tuple  # (lat, lon)
        memory_type: str
        cooling: str
        T2: str
        status: str
        participants: str
        technologies: list = field(default_factory=list)
        node_class: str = "standard"  # standard, hub, sensor

    @dataclass
    class FiberLink:
        """Fiber connection between nodes."""
        start_uid: str
        end_uid: str
        label: str
        length_m: float
        transmission_dB: float
        link_type: str  # fiber, switch/ROADM, satellite, free-space

    return FiberLink, QuantumNode


@app.cell
def _(QuantumNode):
    # BARQNET Node Definitions (Boston-Area Network)
    # From dirkenglund/quantum-network-code with IoCQT extensions

    barqnet_nodes = {
        # MIT Campus Nodes
        "MIT-FH576-A": QuantumNode(
            "MIT-FH576-A", "Fiber Hub 576-A", (42.3619, -71.0903),
            "N/A", "4K", "N/A", "Online", "CQN",
            ["ROADM", "Fiber switching"], "hub"
        ),
        "MIT-FH576-B": QuantumNode(
            "MIT-FH576-B", "Fiber Hub 576-B", (42.3619, -71.0908),
            "N/A", "4K", "N/A", "Online", "CQN",
            ["ROADM", "Fiber switching"], "hub"
        ),
        "MIT-26-465": QuantumNode(
            "MIT-26-465", "Building 26, Room 465", (42.3612, -71.0920),
            "SiV", "0.1K", ">2ms", "Installed", "Englund Lab",
            ["ZALM sources", "QFC", "Nanophotonics"], "standard"
        ),
        "MIT-26-368": QuantumNode(
            "MIT-26-368", "Building 26, Room 368", (42.3612, -71.0925),
            "SiV", "0.3K", ">1ms", "Installed", "Englund Lab",
            ["MiRP receivers", "Diamond sensors"], "sensor"
        ),
        "MIT-36-576": QuantumNode(
            "MIT-36-576", "Building 36, Room 576", (42.3605, -71.0915),
            "SnV", "1K", ">1ms", "Planned", "Englund Lab",
            ["Tin-vacancy memories", "Photonic integration"]
        ),
        "MIT-38-177": QuantumNode(
            "MIT-38-177", "Building 38, Room 177", (42.3600, -71.0930),
            "NV", "RT", ">10us", "Online", "CQN",
            ["NV-cavity magnetometry", "RF sensing"], "sensor"
        ),
        "MIT-38-185": QuantumNode(
            "MIT-38-185", "Building 38, Room 185", (42.3600, -71.0935),
            "NV", "RT", "N/A", "Online", "CQN",
            ["NV ensemble sensing", "Wide-field imaging"], "sensor"
        ),
        "NOTAROS": QuantumNode(
            "NOTAROS", "Notaros Lab", (42.3615, -71.0940),
            "N/A", "RT", "N/A", "Online", "Notaros Lab",
            ["Integrated photonics", "Optical phased arrays"]
        ),

        # Partner Institutions (Boston Area)
        "HARV": QuantumNode(
            "HARV", "Harvard University", (42.3656, -71.1032),
            "SiV", "1K", ">1ms", "Installed", "MIT-LL",
            ["Quantum Frequency Conversion", "Diamond nanophotonics"]
        ),
        "MIT-LL": QuantumNode(
            "MIT-LL", "MIT Lincoln Laboratory", (42.4431, -71.2686),
            "SiV", "0.3K", ">1ms", "Online", "MIT-LL",
            ["Integrated QFC", "Cryogenic packaging"]
        ),
        "BU-QuNETT": QuantumNode(
            "BU-QuNETT", "Boston University QuNETT", (42.3505, -71.1054),
            "NV", "RT", ">10us", "Online", "BU",
            ["NV centers", "Quantum networking testbed"]
        ),
        "BBN": QuantumNode(
            "BBN", "BBN Raytheon", (42.3942, -71.1467),
            "N/A", "RT", "N/A", "Online", "Raytheon",
            ["Classical networking", "QKD protocols"]
        ),
    }

    return (barqnet_nodes,)


@app.cell
def _(QuantumNode):
    # MARQI Node Definitions (Maryland Network)
    # Risk mitigation via second network infrastructure

    marqi_nodes = {
        "UMD": QuantumNode(
            "UMD", "University of Maryland - Main", (38.9856, -76.9426),
            "trapped ions", "RT", ">1s", "Online", "JQI/IonQ",
            ["Trapped-ion qubits", "High-fidelity gates", "Long coherence"]
        ),
        "UMD-NIST": QuantumNode(
            "UMD-NIST", "NIST Gaithersburg", (39.1340, -77.2167),
            "neutral atoms", "uK", ">100ms", "Online", "NIST",
            ["Neutral atom arrays", "Atomic clocks", "Rydberg interactions"]
        ),
        "UMD-ARL": QuantumNode(
            "UMD-ARL", "Army Research Lab", (39.0180, -76.9511),
            "OPM", "RT", "N/A", "Online", "ARL",
            ["Optically pumped magnetometers", "Rydberg RF sensors"]
        ),
        "HU": QuantumNode(
            "HU", "Howard University", (38.9225, -77.0200),
            "N/A", "RT", "N/A", "Planned", "HU",
            ["Workforce development", "HBCU partnership"]
        ),
    }

    return (marqi_nodes,)


@app.cell
def _(FiberLink):
    # Fiber Links - BARQNET Internal
    barqnet_links = [
        FiberLink("MIT-FH576-A", "MIT-FH576-B", "Patch fiber", 10, -0.1, "fiber"),
        FiberLink("MIT-FH576-A", "MIT-26-465", "SMF-28", 200, -0.5, "fiber"),
        FiberLink("MIT-FH576-A", "MIT-26-368", "SMF-28", 250, -0.5, "fiber"),
        FiberLink("MIT-FH576-B", "MIT-36-576", "SMF-28", 300, -0.6, "fiber"),
        FiberLink("MIT-FH576-B", "MIT-38-177", "SMF-28", 400, -0.8, "fiber"),
        FiberLink("MIT-FH576-B", "MIT-38-185", "SMF-28", 420, -0.8, "fiber"),
        FiberLink("MIT-FH576-A", "HARV", "2x SMF-28", 5000, -1.5, "fiber"),
        FiberLink("MIT-FH576-A", "MIT-LL", "SMF-28", 25000, -5.0, "fiber"),
        FiberLink("MIT-FH576-B", "BU-QuNETT", "SMF-28", 3000, -0.8, "fiber"),
        FiberLink("MIT-FH576-A", "BBN", "SMF-28", 8000, -2.0, "fiber"),
        FiberLink("MIT-26-465", "NOTAROS", "SMF-28", 150, -0.3, "fiber"),
    ]

    # Inter-network Link (Boston to MARQI)
    boston_marqi_link = FiberLink(
        "BU-QuNETT", "UMD", "Long-haul SMF-28",
        600000,  # ~600 km
        -120,    # High loss requiring repeaters
        "fiber"
    )

    # MARQI Internal Links
    marqi_links = [
        FiberLink("UMD", "UMD-NIST", "SMF-28", 30000, -6.0, "fiber"),
        FiberLink("UMD", "UMD-ARL", "SMF-28", 5000, -1.0, "fiber"),
        FiberLink("UMD", "HU", "SMF-28", 12000, -2.5, "fiber"),
        FiberLink("UMD-NIST", "UMD-ARL", "SMF-28", 25000, -5.0, "fiber"),
    ]

    return barqnet_links, boston_marqi_link, marqi_links


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Risk Mitigation via MARQI Network

    The **Maryland Quantum Network Infrastructure (MARQI)** serves as a critical risk mitigation strategy:

    ### Primary Risks Addressed

    | Risk | Mitigation via MARQI |
    |------|---------------------|
    | SiV memory scaling challenges | Trapped-ion backup with >1s coherence |
    | Diamond fabrication yield | Mature ion trap technology at IonQ/JQI |
    | Single-site failure | Geographic redundancy (Boston ↔ Maryland) |
    | Platform lock-in | Multi-platform interoperability demonstration |

    ### MARQI Capabilities

    - **Trapped-ion qubits**: Highest-fidelity two-qubit gates (>99.9%)
    - **Neutral atom arrays**: Scalable to 1000+ qubits
    - **Atomic clocks**: NIST primary frequency standards
    - **OPM/Rydberg sensors**: Complementary RF sensing modalities

    ### NSF Convergence Accelerator Funding

    MARQI is funded through the NSF Convergence Accelerator program, ensuring:
    - Dedicated infrastructure investment
    - Multi-institutional coordination
    - Workforce development (HBCU partnerships)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Scaling Goals: SiV Nuclear Spin Qubits

    ### Target: 4800 Quantum Memories

    The long-term vision involves scaling to **4800 quantum memories** using SiV nuclear spin qubits:

    $$
    N_{\text{memories}} = N_{\text{nodes}} \times N_{\text{memories/node}} = 48 \times 100 = 4800
    $$

    ### SiV Nuclear Spin Architecture

    | Parameter | Current | Target (5-year) |
    |-----------|---------|-----------------|
    | Nuclear spin $T_2$ | >10 ms | >100 ms |
    | Electronic-nuclear coupling | 100 kHz | 1 MHz |
    | Gate fidelity | 95% | 99.5% |
    | Memories per node | 1-10 | 100 |
    | Total network memories | ~50 | 4800 |

    ### Key Technical Advances Required

    1. **Nuclear spin initialization**: Optical pumping via SiV electronic transition
    2. **Coherent control**: Microwave/RF driving of nuclear spin states
    3. **Readout**: Electron-nuclear entanglement followed by optical readout
    4. **Scaling**: Photonic integration of multiple SiV centers per chip
    """)
    return


@app.cell
def _(np, plt):
    # Scaling projection visualization
    years = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030])
    memories_projection = np.array([10, 50, 200, 500, 1200, 2800, 4800])

    fig_scaling, ax_scaling = plt.subplots(figsize=(10, 6))
    ax_scaling.semilogy(years, memories_projection, 'o-', linewidth=2, markersize=10, color='#2E86AB')
    ax_scaling.fill_between(years, memories_projection * 0.5, memories_projection * 1.5, alpha=0.2, color='#2E86AB')

    ax_scaling.set_xlabel('Year', fontsize=12)
    ax_scaling.set_ylabel(r'Total Network Memories $N_{\mathrm{mem}}$', fontsize=12)
    ax_scaling.set_title(r'SiV Nuclear Spin Qubit Scaling Roadmap: $N_{\mathrm{mem}} \rightarrow 4800$', fontsize=14)
    ax_scaling.grid(True, alpha=0.3)
    ax_scaling.set_ylim(5, 10000)

    # Annotate key milestones
    ax_scaling.annotate('Current\n(~10 memories)', xy=(2024, 10), xytext=(2024.3, 30),
                       fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    ax_scaling.annotate('Target\n(4800 memories)', xy=(2030, 4800), xytext=(2029, 2000),
                       fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    fig_scaling
    return ax_scaling, fig_scaling, memories_projection, years


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. ZALM-Enabled Quantum Repeaters

    ### Zero-Added-Loss Multiplexing (ZALM)

    ZALM is a key enabling technology for quantum repeaters that allows **multiplexing without additional photon loss**:

    $$
    \eta_{\text{ZALM}} = \eta_{\text{channel}} \times \underbrace{\eta_{\text{mux}}}_{= 1} = \eta_{\text{channel}}
    $$

    Compare to conventional multiplexing:
    $$
    \eta_{\text{conventional}} = \eta_{\text{channel}} \times \eta_{\text{mux}} \approx \eta_{\text{channel}} \times 0.1
    $$

    ### ZALM Implementation at Node 1 (Boston)

    | Component | Specification | Status |
    |-----------|--------------|--------|
    | Entangled photon sources | 1550 nm telecom band | Operational |
    | Temporal multiplexing | 10 GHz repetition rate | In development |
    | Frequency multiplexing | 100 channels × 50 GHz spacing | Planned |
    | Memory interface | SiV spin-photon interface | Demonstrated |

    ### Repeater Architecture

    ```
    Node A ──[ZALM source]──► Channel ──► [ZALM receiver]──► Node B
                    │                              │
                    ▼                              ▼
              [SiV Memory]                   [SiV Memory]
                    │                              │
                    └──────── Entanglement ────────┘
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Two-Node Network Architecture

    ### Node 1: Boston (MIT/CQN)

    **Primary capabilities:**
    - **SiV quantum memories**: Nuclear spin qubits with >2 ms $T_2$
    - **NV-cavity RF/magnetometry**: Room-temperature quantum sensing
    - **MiRP receivers**: Microwave-infrared-photonic transduction
    - **ZALM entangled-photon sources**: High-rate, low-loss multiplexing

    ### Node 2: MARQI (UMD)

    **Primary capabilities:**
    - **Trapped-ion nodes**: >1 s coherence, highest gate fidelities
    - **Neutral-atom nodes**: Scalable arrays with Rydberg interactions
    - **Atomic clocks**: NIST-traceable frequency standards
    - **OPM/Rydberg RF sensors**: Broadband electromagnetic sensing

    ### Inter-Node Connection

    The two nodes are connected via:
    1. **Direct fiber**: ~600 km requiring quantum repeaters
    2. **Classical channel**: High-bandwidth for CQML data fusion
    3. **Timing synchronization**: GPS-disciplined atomic clocks
    """)
    return


@app.cell
def _(barqnet_nodes, marqi_nodes, nx, plt):
    # Create network graph visualization
    G = nx.Graph()

    # Add BARQNET nodes
    for uid, node in barqnet_nodes.items():
        G.add_node(uid,
                   name=node.name,
                   network='BARQNET',
                   memory=node.memory_type,
                   color='#2E86AB' if node.node_class == 'standard' else
                         '#A23B72' if node.node_class == 'hub' else '#F18F01')

    # Add MARQI nodes
    for uid, node in marqi_nodes.items():
        G.add_node(uid,
                   name=node.name,
                   network='MARQI',
                   memory=node.memory_type,
                   color='#C73E1D')

    # Add edges (simplified for visualization)
    barqnet_edges = [
        ("MIT-FH576-A", "MIT-FH576-B"),
        ("MIT-FH576-A", "MIT-26-465"),
        ("MIT-FH576-A", "MIT-26-368"),
        ("MIT-FH576-B", "MIT-36-576"),
        ("MIT-FH576-B", "MIT-38-177"),
        ("MIT-FH576-B", "MIT-38-185"),
        ("MIT-FH576-A", "HARV"),
        ("MIT-FH576-A", "MIT-LL"),
        ("MIT-FH576-B", "BU-QuNETT"),
        ("MIT-FH576-A", "BBN"),
        ("MIT-26-465", "NOTAROS"),
    ]

    marqi_edges = [
        ("UMD", "UMD-NIST"),
        ("UMD", "UMD-ARL"),
        ("UMD", "HU"),
    ]

    inter_network_edge = [("BU-QuNETT", "UMD")]

    G.add_edges_from(barqnet_edges, network='BARQNET', style='solid')
    G.add_edges_from(marqi_edges, network='MARQI', style='solid')
    G.add_edges_from(inter_network_edge, network='inter', style='dashed')

    # Create visualization
    fig_network, ax_network = plt.subplots(figsize=(14, 10))

    # Position nodes using spring layout with some manual adjustments
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Separate BARQNET and MARQI for better visualization
    for node in barqnet_nodes:
        if node in pos:
            pos[node] = (pos[node][0] - 0.5, pos[node][1] + 0.3)
    for node in marqi_nodes:
        if node in pos:
            pos[node] = (pos[node][0] + 0.5, pos[node][1] - 0.5)

    # Draw nodes by type
    node_colors = [G.nodes[n].get('color', '#888888') for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9, ax=ax_network)

    # Draw edges
    barqnet_edge_list = [(u, v) for u, v in G.edges() if G.edges[u, v].get('network') == 'BARQNET']
    marqi_edge_list = [(u, v) for u, v in G.edges() if G.edges[u, v].get('network') == 'MARQI']
    inter_edge_list = [(u, v) for u, v in G.edges() if G.edges[u, v].get('network') == 'inter']

    nx.draw_networkx_edges(G, pos, edgelist=barqnet_edge_list, edge_color='#2E86AB', width=2, alpha=0.7, ax=ax_network)
    nx.draw_networkx_edges(G, pos, edgelist=marqi_edge_list, edge_color='#C73E1D', width=2, alpha=0.7, ax=ax_network)
    nx.draw_networkx_edges(G, pos, edgelist=inter_edge_list, edge_color='#333333', width=3, style='dashed', alpha=0.8, ax=ax_network)

    # Labels
    labels = {n: n.replace('MIT-', '').replace('UMD-', '') for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax_network)

    ax_network.set_title('Two-Node Quantum Network: BARQNET (Boston) ↔ MARQI (Maryland)', fontsize=14)
    ax_network.axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='BARQNET Standard'),
        Patch(facecolor='#A23B72', label='BARQNET Hub'),
        Patch(facecolor='#F18F01', label='BARQNET Sensor'),
        Patch(facecolor='#C73E1D', label='MARQI Node'),
    ]
    ax_network.legend(handles=legend_elements, loc='lower left', fontsize=9)

    plt.tight_layout()
    fig_network
    return (
        G,
        Patch,
        ax_network,
        barqnet_edge_list,
        barqnet_edges,
        fig_network,
        inter_edge_list,
        inter_network_edge,
        labels,
        legend_elements,
        marqi_edge_list,
        marqi_edges,
        node_colors,
        pos,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Node 1 (Boston) Detailed Specifications

    ### 6.1 SiV Quantum Memories

    Silicon-vacancy (SiV) centers in diamond serve as the primary quantum memory platform:

    | Property | Value | Notes |
    |----------|-------|-------|
    | Optical transition | 737 nm | Zero-phonon line |
    | Telecom conversion | 1550 nm via QFC | Quantum frequency conversion |
    | Electronic spin $T_2$ | ~10 ms @ 100 mK | Dilution refrigerator required |
    | Nuclear spin $T_2$ | >100 ms | $^{13}$C or $^{29}$Si nuclear spins |
    | Spin-photon entanglement fidelity | >95% | Demonstrated |
    | Purcell enhancement | >100× | Nanophotonic cavity |

    ### 6.2 NV-Cavity RF/Magnetometry

    Nitrogen-vacancy (NV) centers provide room-temperature quantum sensing:

    $$
    \eta_{\text{NV}} = \frac{\gamma_e \cdot C \cdot \sqrt{n}}{\sqrt{T_2^* \cdot \text{BW}}}
    $$

    where $\gamma_e = 28$ GHz/T is the electron gyromagnetic ratio, $C$ is contrast, $n$ is photon collection efficiency.

    | Sensing Modality | Sensitivity | Bandwidth |
    |-----------------|-------------|-----------|
    | DC magnetometry | 1 pT/√Hz | DC-100 Hz |
    | AC magnetometry | 100 fT/√Hz | 1 kHz - 1 MHz |
    | RF sensing (MiRP) | -140 dBm/Hz | 1 MHz - 10 GHz |
    | Temperature | 10 mK/√Hz | DC-1 kHz |

    ### 6.3 MiRP Receivers

    Microwave-infrared-photonic (MiRP) transduction enables RF signal detection:

    $$
    \text{NEP}_{\text{MiRP}} = \frac{h\nu}{\eta_{\text{QE}} \cdot \sqrt{\tau}} \approx 10^{-22} \text{ W/√Hz}
    $$

    ### 6.4 ZALM Entangled-Photon Sources

    | Parameter | Specification |
    |-----------|--------------|
    | Photon pair rate | $10^9$ pairs/s |
    | Heralding efficiency | >90% |
    | Spectral brightness | $10^6$ pairs/s/mW/GHz |
    | Entanglement fidelity | >98% |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Node 2 (MARQI/UMD) Detailed Specifications

    ### 7.1 Trapped-Ion Nodes

    Trapped-ion systems at JQI/IonQ provide the highest-fidelity quantum operations:

    | Property | Value | Notes |
    |----------|-------|-------|
    | Ion species | $^{171}$Yb$^+$, $^{138}$Ba$^+$ | Hyperfine qubits |
    | Qubit coherence $T_2$ | >1 s | Magnetic field shielding |
    | Single-qubit gate fidelity | 99.99% | Demonstrated |
    | Two-qubit gate fidelity | 99.9% | Mølmer-Sørensen gates |
    | Ion-photon entanglement | 95% fidelity | Cavity-enhanced |
    | Telecom conversion | 1550 nm via QFC | For long-distance links |

    ### 7.2 Neutral-Atom Nodes

    Neutral atom arrays at NIST enable scalable quantum processing:

    $$
    H_{\text{Rydberg}} = \sum_i \Omega_i \sigma_i^x + \sum_i \Delta_i n_i + \sum_{i<j} V_{ij} n_i n_j
    $$

    where $V_{ij} = C_6/r_{ij}^6$ is the Rydberg interaction.

    | Property | Value |
    |----------|-------|
    | Array size | 100-1000 atoms |
    | Rydberg interaction range | ~10 μm |
    | Gate time | ~100 ns |
    | Atom loss rate | <0.1% per operation |

    ### 7.3 Atomic Clocks

    NIST atomic clocks provide network-wide timing synchronization:

    | Clock Type | Fractional Stability | Role |
    |------------|---------------------|------|
    | Optical lattice (Sr) | $10^{-18}$ | Primary reference |
    | Ion clock (Al$^+$) | $10^{-18}$ | Verification |
    | Cs fountain | $10^{-16}$ | SI second realization |
    | Portable (Rb) | $10^{-13}$ | Network distribution |

    ### 7.4 OPM/Rydberg RF Sensors

    | Sensor Type | Sensitivity | Bandwidth | Notes |
    |-------------|-------------|-----------|-------|
    | OPM (K, Rb) | 1 fT/√Hz | DC-1 kHz | Biomagnetic sensing |
    | Rydberg EIT | $\mu$V/m/√Hz | 1 MHz - 100 GHz | SI-traceable RF |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Sensor Integration Across Nodes

    ### Cross-Platform Sensor Fusion

    The two-node architecture enables **heterogeneous sensor fusion** combining:

    | Sensor Type | Location | Measurement | Integration |
    |-------------|----------|-------------|-------------|
    | Diamond SiV | Boston | Single-spin magnetic fields | Quantum memory readout |
    | NV ensemble | Boston | Wide-field magnetometry | Classical image sensor |
    | MiRP | Boston | RF spectrum | Photonic readout |
    | OPM | MARQI | Biomagnetic fields | MEG/MCG arrays |
    | Rydberg | MARQI | RF electric fields | SI-traceable calibration |
    | Atomic clock | MARQI | Time/frequency | Network synchronization |

    ### Data Fusion Architecture

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    CQML Data Fusion Layer                   │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
    │  │ Diamond SiV │    │   NV-Cavity │    │    MiRP     │     │
    │  │ (quantum)   │    │ (classical) │    │ (classical) │     │
    │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
    │         │                  │                  │             │
    │         └──────────────────┼──────────────────┘             │
    │                            │                                │
    │                     ┌──────▼──────┐                         │
    │                     │   Node 1    │                         │
    │                     │  (Boston)   │                         │
    │                     └──────┬──────┘                         │
    │                            │ Fiber/Classical                │
    │                     ┌──────▼──────┐                         │
    │                     │   Node 2    │                         │
    │                     │  (MARQI)    │                         │
    │                     └──────┬──────┘                         │
    │                            │                                │
    │         ┌──────────────────┼──────────────────┐             │
    │         │                  │                  │             │
    │  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐     │
    │  │ Trapped Ion │    │   Rydberg   │    │ Atomic Clock│     │
    │  │  (quantum)  │    │ (classical) │    │ (classical) │     │
    │  └─────────────┘    └─────────────┘    └─────────────┘     │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Digital Twin Framework for CQML

    ### Architecture Overview

    The **Digital Twin** provides a differentiable model for hardware-in-the-loop calibration:

    $$
    \mathcal{L}(\theta) = \sum_i \left\| y_i^{\text{measured}} - f_\theta(x_i) \right\|^2 + \lambda R(\theta)
    $$

    where:
    - $f_\theta$: Differentiable physics model (JAX/PyTorch)
    - $y_i^{\text{measured}}$: Real sensor measurements
    - $R(\theta)$: Regularization (physics constraints)
    - $\lambda$: Regularization strength

    ### Components

    | Component | Implementation | Latency |
    |-----------|---------------|---------|
    | Quantum state tomography | JAX autodiff | <1 ms |
    | Error model | Pauli channel estimation | <0.5 ms |
    | Classical sensor fusion | PyTorch neural network | <0.1 ms |
    | Feedback control | Real-time optimizer | <0.3 ms |
    | **Total loop latency** | | **<1 ms** |

    ### Differentiable Quantum Simulation

    The digital twin uses **differentiable quantum circuits** for gradient-based optimization:

    ```python
    # Pseudocode for digital twin update
    def digital_twin_update(params, measurements):
        # Forward pass: simulate quantum system
        predicted = quantum_circuit(params)

        # Compute loss
        loss = mse_loss(predicted, measurements)

        # Backward pass: compute gradients
        grads = jax.grad(loss)(params)

        # Update parameters
        new_params = params - learning_rate * grads

        return new_params
    ```
    """)
    return


@app.cell
def _(np, plt):
    # Digital Twin latency breakdown visualization
    components = ['Quantum\nTomography', 'Error\nModel', 'Sensor\nFusion', 'Feedback\nControl']
    latencies = [0.45, 0.25, 0.1, 0.15]  # ms
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    fig_latency, ax_latency = plt.subplots(figsize=(10, 5))
    bars = ax_latency.bar(components, latencies, color=colors, edgecolor='black', linewidth=1.5)

    ax_latency.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Target: <1 ms total')
    ax_latency.set_ylabel('Latency (ms)', fontsize=12)
    ax_latency.set_title(r'Digital Twin Loop Latency: $\tau_{\mathrm{total}} < 1$ ms', fontsize=14)
    ax_latency.set_ylim(0, 1.2)

    # Annotate total
    total_latency = sum(latencies)
    ax_latency.annotate(f'Total: {total_latency:.2f} ms', xy=(2.5, total_latency + 0.1),
                       fontsize=11, fontweight='bold', ha='center')

    # Add value labels on bars
    for _bar, _val in zip(bars, latencies):
        ax_latency.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 0.02,
                       f'{_val:.2f} ms', ha='center', va='bottom', fontsize=10)

    ax_latency.legend(loc='upper right')
    plt.tight_layout()
    fig_latency
    return (
        ax_latency,
        bars,
        colors,
        components,
        fig_latency,
        latencies,
        total_latency,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Hardware-in-the-Loop Calibration

    ### Calibration Protocol

    The <1 ms latency requirement enables **real-time feedback** for:

    1. **Quantum state preparation**: Correct initialization errors
    2. **Gate calibration**: Compensate for drift in pulse parameters
    3. **Measurement basis alignment**: Optimize readout fidelity
    4. **Environmental compensation**: Cancel magnetic field fluctuations

    ### Feedback Loop Architecture

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    Hardware-in-the-Loop                     │
    │                                                             │
    │   ┌─────────┐      ┌─────────┐      ┌─────────┐            │
    │   │ Quantum │──────│ Digital │──────│ Control │            │
    │   │ Hardware│ meas │  Twin   │ ctrl │ Hardware│            │
    │   └────┬────┘      └────┬────┘      └────┬────┘            │
    │        │                │                │                  │
    │        │    <0.5 ms     │    <0.5 ms     │                  │
    │        └────────────────┴────────────────┘                  │
    │                    Total: <1 ms                             │
    └─────────────────────────────────────────────────────────────┘
    ```

    ### Latency Budget

    | Stage | Allocated | Achieved |
    |-------|-----------|----------|
    | Measurement acquisition | 200 μs | 150 μs |
    | Data transfer to GPU | 50 μs | 30 μs |
    | Digital twin inference | 500 μs | 450 μs |
    | Gradient computation | 100 μs | 80 μs |
    | Control signal generation | 100 μs | 90 μs |
    | Signal upload to AWG | 50 μs | 40 μs |
    | **Total** | **1000 μs** | **840 μs** |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Physically-Guided Feature Extraction

    ### Motivation: Latency Reduction via Physics

    Instead of generic neural network feature extraction, we use **physics-informed features** that:

    1. Reduce dimensionality by 10-100×
    2. Preserve physically relevant information
    3. Enable faster inference (latency reduction)
    4. Improve generalization to unseen conditions

    ### Feature Categories

    | Feature Type | Description | Dimension |
    |--------------|-------------|-----------|
    | **Bloch vector** | $\vec{r} = (\langle\sigma_x\rangle, \langle\sigma_y\rangle, \langle\sigma_z\rangle)$ | 3 |
    | **Purity** | $\gamma = \text{Tr}(\rho^2)$ | 1 |
    | **Entanglement entropy** | $S = -\text{Tr}(\rho \log \rho)$ | 1 |
    | **Fidelity** | $F = \langle\psi_{\text{target}}|\rho|\psi_{\text{target}}\rangle$ | 1 |
    | **Error channel params** | $(p_x, p_y, p_z, p_I)$ for Pauli channel | 4 |

    ### Mathematical Framework

    For a quantum state $\rho$ and target $|\psi_{\text{target}}\rangle$:

    $$
    \text{Features}(\rho) = \begin{pmatrix}
    \text{Tr}(\sigma_x \rho) \\
    \text{Tr}(\sigma_y \rho) \\
    \text{Tr}(\sigma_z \rho) \\
    \text{Tr}(\rho^2) \\
    F(\rho, |\psi_{\text{target}}\rangle)
    \end{pmatrix}
    $$

    ### Latency Comparison

    | Approach | Feature Dim | Inference Time |
    |----------|-------------|----------------|
    | Raw density matrix | $4^n$ | 100 ms |
    | CNN features | 512 | 10 ms |
    | Physics-guided | 5-10 | **<1 ms** |
    """)
    return


@app.cell
def _(np, plt):
    # Feature extraction latency comparison
    approaches = ['Raw Density\nMatrix', 'CNN\nFeatures', 'Physics-Guided\nFeatures']
    inference_times = [100, 10, 0.8]  # ms
    feature_dims = [256, 512, 8]

    fig_features, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Inference time comparison
    colors_feat = ['#888888', '#A23B72', '#2E86AB']
    bars1 = ax1.bar(approaches, inference_times, color=colors_feat, edgecolor='black')
    ax1.set_ylabel('Inference Time (ms)', fontsize=12)
    ax1.set_title('Latency Comparison: Feature Extraction Approaches', fontsize=12)
    ax1.set_yscale('log')
    ax1.axhline(y=1.0, color='red', linestyle='--', label='Target: <1 ms')
    ax1.legend()

    for _b, _v in zip(bars1, inference_times):
        ax1.text(_b.get_x() + _b.get_width()/2, _b.get_height() * 1.2,
                f'{_v} ms', ha='center', va='bottom', fontsize=10)

    # Feature dimension comparison
    bars2 = ax2.bar(approaches, feature_dims, color=colors_feat, edgecolor='black')
    ax2.set_ylabel('Feature Dimension', fontsize=12)
    ax2.set_title('Dimensionality: Physics Reduces Complexity', fontsize=12)
    ax2.set_yscale('log')

    for _b2, _v2 in zip(bars2, feature_dims):
        ax2.text(_b2.get_x() + _b2.get_width()/2, _b2.get_height() * 1.2,
                f'{_v2}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig_features
    return (
        approaches,
        ax1,
        ax2,
        bars1,
        bars2,
        colors_feat,
        feature_dims,
        fig_features,
        inference_times,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. CQML Algorithm Architecture

    ### Classical-Quantum Machine Learning Pipeline

    The CQML architecture combines classical neural networks with quantum processing:

    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        CQML Pipeline                                │
    │                                                                     │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
    │  │Classical │    │ Physics  │    │ Quantum  │    │Classical │     │
    │  │ Encoder  │───▶│ Feature  │───▶│   PQC    │───▶│ Decoder  │     │
    │  │  (CNN)   │    │Extraction│    │          │    │  (MLP)   │     │
    │  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
    │       │               │               │               │            │
    │       ▼               ▼               ▼               ▼            │
    │   Raw data      Physics-guided   Entangled       Predictions      │
    │   (images,      features         quantum          (labels,        │
    │    signals)     (Bloch, purity)  features         values)         │
    └─────────────────────────────────────────────────────────────────────┘
    ```

    ### Parameterized Quantum Circuit (PQC)

    The quantum component uses a variational ansatz:

    $$
    U(\theta) = \prod_{l=1}^{L} \left[ \prod_{i} R_Y(\theta_{l,i}) \cdot \prod_{(i,j)} \text{CNOT}_{ij} \right]
    $$

    ### Training Objective

    Joint optimization of classical ($\phi$) and quantum ($\theta$) parameters:

    $$
    \min_{\phi, \theta} \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \mathcal{L}\left( f_\phi \circ U_\theta \circ g_\phi(x), y \right) \right]
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Summary and Roadmap

    ### Key Achievements (Current)

    | Capability | Status | TRL |
    |------------|--------|-----|
    | BARQNET fiber infrastructure | Operational | 7 |
    | SiV quantum memories | Demonstrated | 4 |
    | NV magnetometry | Operational | 6 |
    | Trapped-ion nodes (MARQI) | Operational | 6 |
    | Digital Twin framework | In development | 3 |

    ### Near-Term Goals (1-2 years)

    1. **Entanglement distribution**: Demonstrate SiV-SiV entanglement across 50 km
    2. **ZALM deployment**: 100× multiplexing with zero added loss
    3. **Cross-platform teleportation**: SiV (Boston) ↔ Ion (MARQI)
    4. **Digital Twin v1.0**: <1 ms calibration loop operational

    ### Long-Term Vision (5+ years)

    1. **4800 quantum memories**: Full network deployment
    2. **Continental scale**: Boston-Maryland-Chicago quantum backbone
    3. **IoCQT applications**: Distributed quantum sensing, CQML, quantum internet
    4. **Quantum advantage**: Demonstrated for specific sensing/ML tasks

    ---

    ### References

    1. BARQNET infrastructure: [MIT CQN](https://cqn.mit.edu)
    2. MARQI network: [NSF Convergence Accelerator](https://nsf.gov/convergence)
    3. SiV memories: Bhaskar et al., Nature (2020)
    4. Trapped-ion networking: Monroe et al., Rev. Mod. Phys. (2021)
    5. ZALM multiplexing: Sinclair et al., Phys. Rev. Lett. (2023)
    6. Digital Twin for quantum: [quantum-network-code](https://github.com/dirkenglund/quantum-network-code)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Document prepared for ROAD-IoCQT STC**
    *Boston-Area Quantum Network and Maryland Quantum Infrastructure*

    For questions contact: [STC-IoCQT](mailto:stc-iocqt@mit.edu)
    """)
    return


if __name__ == "__main__":
    app.run()
