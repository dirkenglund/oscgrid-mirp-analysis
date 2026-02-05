# OscGrid Dataset: Photonic & Quantum Approaches for Power Grid Analysis

**Date:** 2026-02-04
**Dataset:** OscGrid (Open Dataset of Real-World Oscillograms from Electric Power Grids)
**Source:** [Figshare](https://figshare.com/articles/dataset/28465427)

---

## Executive Summary

This analysis evaluates the OscGrid power grid dataset for compatibility with advanced photonic and quantum sensing/processing approaches. While the dataset is **not suitable** for Quantum Search Sensing (QSS) from arXiv:2501.07625v1 (Choi et al.), it presents excellent opportunities for:

1. **Photonic Perceptron-based Fault Detection** (high compatibility)
2. **Entanglement-Enhanced Sensing for Precision Measurement** (medium compatibility)
3. **Distributed Quantum Sensor Networks** (medium compatibility)
4. **Classical-Quantum Hybrid Anomaly Detection** (high compatibility)

---

## 1. OscGrid Dataset Characterization

### Data Structure
```
Path: /CSV_format_v1.1/labeled_sample/labeled.csv
Size: 384 MB (labeled sample)
Total: 50,765 oscillograms
```

### Signal Columns
| Column | Description | Units | Typical Range |
|--------|-------------|-------|---------------|
| IA, IB, IC | Phase currents (A, B, C) | Amperes | 0.01 - 0.05 |
| IN | Neutral current | Amperes | ~0.03 |
| UA_BB, UB_BB, UC_BB | Bus bar voltages | Volts | -80 to +80 |
| UA_CL, UB_CL, UC_CL | Cable voltages | Volts | -80 to +80 |
| UAB_*, UBC_*, UCA_* | Line-to-line voltages | Volts | Various |

### Classification Labels
- `ML_1` through `ML_6`: Fault type classifications
- Labeled for supervised machine learning training

### Key Characteristics
- **Frequency**: Known (50 Hz or 60 Hz depending on grid)
- **Signal Strength**: Strong (mA-A currents, 10s-100s V)
- **Sensor Type**: Classical (current transformers, voltage dividers)
- **Time Series**: Oscillogram format with temporal structure

---

## 2. Why NOT Quantum Search Sensing (QSS)?

The Choi paper (arXiv:2501.07625v1) QSS algorithm requires:

| QSS Requirement | OscGrid Reality | Match? |
|-----------------|-----------------|--------|
| Unknown frequency to search | Known 50/60 Hz | ❌ |
| Weak signals (nT-pT magnetic) | Strong signals (mA-A) | ❌ |
| Quantum sensors (NV centers) | Classical sensors | ❌ |
| Grover search speedup needed | No search problem | ❌ |

**Conclusion**: QSS provides Grover speedup Ω(√N) only when searching for unknown frequencies. With known grid frequency, there's no search problem, hence no quantum advantage from QSS.

---

## 3. Photonic Perceptron for Power Grid Fault Detection

### Technology Overview (Mancinelli et al., Scientific Reports)

The photonic complex perceptron is a silicon photonics integrated circuit that:
- Processes data at **16 Gbps** (ultrafast)
- Uses **delay lines + phase modulators** for temporal pattern recognition
- Achieves BER as low as **10⁻⁶** for classification tasks
- Trained via **Particle Swarm Optimization** on phase values

### Architecture
```
Input u(t) → 1×4 Splitter → [Delay Lines τ₁-τ₄] → [Phase Modulators φ₁-φ₄] → 4×1 Combiner → Photodetector → y(t)

Output: y(t) = |Σₖ wₖ · uₖ(t)|²
where wₖ = exp(iφₖ) are trainable complex weights
```

### OscGrid Compatibility Analysis

#### ✅ High Compatibility Factors

1. **Time-Series Nature**
   - OscGrid oscillograms are inherently time-series
   - Photonic perceptron excels at temporal pattern recognition
   - Delay-line architecture naturally processes sequential samples

2. **Pattern Recognition Task**
   - OscGrid includes ML labels for fault classification
   - Perceptron demonstrated 2-3 bit pattern recognition
   - Fault signatures in power grid waveforms are pattern-based

3. **Multi-Phase Processing**
   - 4 delay lines match 3-phase + neutral (IA, IB, IC, IN)
   - Can extend to larger splitter for voltage channels

4. **Ultrafast Processing**
   - 16 Gbps allows real-time grid monitoring
   - Could process thousands of oscillograms per second

#### ⚠️ Integration Considerations

1. **Analog-to-Optical Conversion**
   - OscGrid data is digital (CSV)
   - For real-time use: need electro-optic modulators to encode current/voltage
   - For offline training: can simulate optical processing digitally

2. **Sampling Rate Matching**
   - Photonic perceptron: ~50 ps delay resolution
   - Power grid: 50/60 Hz fundamental (~16-20 ms period)
   - Need proper downsampling or multi-scale processing

3. **Training Methodology**
   - Use OscGrid labeled data to optimize phase values φₖ
   - PSO algorithm compatible with differentiable optical simulation

### Proposed Pipeline

```
OscGrid CSV → Preprocessing → [Analog Encoding] → Photonic Perceptron → Fault Classification
                    ↓
           Digital Twin (Simulation)
                    ↓
           Phase Optimization via PSO
```

### Expected Benefits
- **Latency**: Sub-microsecond inference (vs. ms for digital)
- **Energy**: ~fJ/op (vs. pJ for digital neural networks)
- **Throughput**: Parallelizable across multiple devices

---

## 4. Entanglement-Enhanced Sensing (Known Frequency)

### Concept
When frequency is **known** (50/60 Hz), quantum advantage comes from:
- **Squeezed states** reducing measurement noise below shot noise limit
- **Entanglement** between multiple sensor nodes for correlated measurements
- **Heisenberg limit** scaling: σ ∝ 1/N vs. shot-noise limit σ ∝ 1/√N

### Application to Power Grid Monitoring

#### Precision Current Sensing
- Use NV center magnetometry or SQUIDs for ultra-precise current measurement
- Squeezed light interferometry for voltage measurement
- Could detect subtle fault precursors invisible to classical sensors

#### OscGrid as Training/Validation Data
- Classical OscGrid data provides ground truth waveforms
- Train quantum sensors to detect deviations from normal patterns
- Validate quantum sensor outputs against classical measurements

### Compatibility
- **Medium**: OscGrid provides labeled fault data for algorithm development
- Not direct input to quantum sensors, but enables:
  - Fault pattern libraries
  - Detection threshold optimization
  - False positive/negative characterization

---

## 5. Distributed Quantum Sensor Networks

### Concept
Network of quantum sensors (NV centers, atomic clocks) across power grid:
- **Correlated measurements** detect grid-wide events
- **Entanglement distribution** enables beyond-classical coordination
- **Quantum communication** provides secure telemetry

### OscGrid Application

#### Correlation Analysis
```python
# Pseudo-code for distributed sensor analysis
correlations = compute_cross_correlation(
    sensor_A=oscgrid['UA_BB'],
    sensor_B=oscgrid['UA_CL'],
    lags=range(-100, 100)
)
# High correlation at specific lag → fault propagation signature
```

#### Multi-Node Fault Localization
- OscGrid contains bus bar (BB) and cable (CL) measurements
- Treat as distributed sensor network proxy
- Develop correlation-based fault localization algorithms

### Compatibility
- **Medium**: OscGrid BB/CL distinction simulates distributed sensing
- Algorithms developed on OscGrid transferable to quantum sensor networks

---

## 6. Hybrid Classical-Quantum Anomaly Detection

### Architecture
```
Classical Sensors → OscGrid-type data → Photonic Perceptron → Feature Extraction
                                                                      ↓
                                                              Anomaly Score
                                                                      ↓
Quantum Sensors → High-precision measurements → Quantum Verification
                                                                      ↓
                                                              Fault Classification
```

### Two-Stage Detection

1. **Stage 1: Fast Classical/Photonic Screening**
   - Photonic perceptron processes grid data at GHz rates
   - Flags potential anomalies based on OscGrid-trained patterns
   - Low latency, handles normal operation efficiently

2. **Stage 2: Quantum Precision Verification**
   - Quantum sensors measure flagged events with Heisenberg-limited precision
   - Distinguishes real faults from noise/transients
   - Provides detailed fault characterization

### Benefits
- Best of both worlds: speed (photonic) + precision (quantum)
- OscGrid enables training of Stage 1 classifier
- Reduces quantum sensor duty cycle (only on flagged events)

---

## 7. Implementation Roadmap

### Phase 1: Digital Simulation (Weeks 1-4)
- [ ] Build photonic perceptron digital twin in PyTorch
- [ ] Train on OscGrid labeled data
- [ ] Benchmark against classical ML (CNN, LSTM)
- [ ] Publish accuracy/speed comparison

### Phase 2: Correlation Analysis (Weeks 5-8)
- [ ] Implement multi-channel correlation on OscGrid BB/CL data
- [ ] Develop fault propagation detection algorithm
- [ ] Map to distributed sensor network abstraction

### Phase 3: Hardware Integration (Weeks 9-16)
- [ ] Interface with silicon photonics testbed (if available)
- [ ] Validate digital twin against hardware measurements
- [ ] Characterize actual processing speed/energy

### Phase 4: Quantum Sensor Integration (Months 4-6)
- [ ] Define quantum sensor specifications for grid monitoring
- [ ] Design hybrid classical-quantum pipeline
- [ ] Simulate entanglement-enhanced detection

---

## 8. Connections to Englund Group Work

### Relevant Prior Work
- **Photonic reservoir computing** with diamond NV centers
- **Optical neural networks** for pattern recognition
- **Quantum sensing** with nitrogen-vacancy centers

### Integration Opportunities
- NV center magnetometry for current sensing
- Diamond photonics for integrated perceptron + sensor
- Quantum memory for correlation storage

---

## 9. Conclusions

| Approach | OscGrid Compatibility | Quantum Advantage | Recommendation |
|----------|----------------------|-------------------|----------------|
| QSS (Choi paper) | ❌ Low | Grover √N speedup | Not applicable |
| Photonic Perceptron | ✅ High | Speed + Energy | **Pursue first** |
| Entanglement Sensing | ⚠️ Medium | Heisenberg limit | Training/validation |
| Distributed Networks | ⚠️ Medium | Correlated detection | Algorithm development |
| Hybrid Detection | ✅ High | Combined benefits | **Ultimate goal** |

### Next Steps
1. Implement photonic perceptron digital twin on OscGrid data
2. Benchmark fault detection accuracy vs. classical baselines
3. Explore BB/CL correlation for distributed sensing proxy
4. Design hybrid pipeline architecture

---

## References

1. Choi et al., "Quantum Search Sensing," arXiv:2501.07625v1
2. Mancinelli et al., "A photonic complex perceptron for ultrafast data processing," Scientific Reports (2022)
3. OscGrid Dataset, Figshare (10.6084/m9.figshare.28465427)
4. Englund Group, MIT - Photonic systems and quantum sensing publications

---

*Generated by Claude Code analysis of OscGrid dataset compatibility with photonic/quantum approaches*
