import json

# Load the four prompt-response pairs (few-shot examples)
# Here, we simulate with example placeholders for assistant responses.
# In practice, you should replace the values in assistant_responses with your real handcrafted outputs.

# Load prompt JSONs (assuming they're already in Python dicts)
with open("prompt_1.json") as f1, open("prompt_2.json") as f2:
    prompt_1 = json.load(f1)
    prompt_2 = json.load(f2)

# Load final query instance (e.g., prompt_3 as the target)
with open("prompt_6.json") as f3:
    query_prompt = json.load(f3)

# These are your handcrafted, high-quality responses
assistant_responses = [
    "Anomaly Summary: "\
    "The model classified this sequence as normal, reflecting balanced and efficient network behavior over three timesteps of normalized input features processed by an LSTM model."\
    "- Uplink and downlink activity are both high and symmetrical. `ul_tx` and `dl_tx` values range from ~0.92 to 0.98, well above normal and attack means, but balanced across directions — this symmetry is reinforced by LIME, where `dl_tx_t-2`, `dl_tx_t-1`, and `dl_tx_t-0` contribute strongly to the normal prediction (–0.2731, –0.1831, –0.1083)."\
    "- Throughput metrics are elevated but not indicative of attack. `ul_bitrate` values (~0.61) are above normal (0.08) and attack (0.09) means but are paired with high `dl_bitrate` (~0.098), which is over 6× the normal mean and far above the attack average (0.0006). LIME assigns negative contributions to all `dl_bitrate` timesteps (e.g., –0.0902 at T2), supporting the normal classification."\
    "- Retransmissions are low and stable. `ul_retx` and `dl_retx` stay between 0.004 and 0.012, well below typical attack means (0.163 and 0.0541). LIME shows these features decreasing anomaly risk (e.g., `ul_retx_t-0`: –0.0214)."\
    "- Signal quality is strong. `epre` values (~0.74) exceed both normal (0.57) and attack (0.48) means. `pusch_snr` remains above 0.54, and `cqi` is at 1.0 across all steps. SHAP shows these features either neutral or mildly supportive of normal behavior."\
    "- Error rates are zero (`ul_err`, `dl_err` = 0), ruling out radio link instability."\
    "- SHAP local values confirm strong negative contributions from `dl_tx` at every timestep (e.g., –0.0547 at T0, –0.0384 at T2), and modest positive impact from `ul_tx` (e.g., +0.0617 at T2), preserving the balance. Global SHAP importance supports this interpretation — the most influential features (`ul_tx`, `dl_tx`, `dl_bitrate`) behave in line with known normal patterns."\
    "Misclassification Likelihood"\
    "Very low. All high-throughput features show symmetry and efficiency, with no signs of retransmission spikes or traffic asymmetry. Signal conditions are excellent, and LIME/SHAP contributions consistently align with normal behavior."\
    "Recommended Mitigation Steps"\
    "- No action needed. This flow reflects well-functioning, high-throughput activity."\
    "- Optionally log as a reference case for tuning thresholds or validating model reliability."\
    "- Continue monitoring for changes over time or shifts in traffic symmetry."\
,
    "Anomaly Summary: "\
    "The model classified this sequence as anomalous, largely due to persistent high uplink activity and low downlink usage, a pattern typical of DDoS traffic."\
    "The decision was based on three timesteps of normalized network features processed by an LSTM sequence model."\
    "- All timesteps show elevated ul_bitrate values around 0.43, which is over 3.5× the normal mean (0.08) and even above the attack mean (0.09)."\
    "- ul_retx values are consistently high (~0.34), more than double the normal mean (0.086) and aligned with the attack mean (0.16)."\
    "- LIME shows:"\
    "- ul_bitrate_t-0 > 0.22: strong positive contribution of +0.0966."\
    "- ul_retx_t-0 > 0.30: adds another +0.0791 to the anomaly score."\
    "- ul_tx_t-0 and dl_tx_t-0 also contribute positively (+0.0473, +0.0558), reinforcing the uplink-dominant pattern."\
    "- SHAP local values confirm:"\
    "- ul_bitrate_t2: largest SHAP contribution (+0.1649), followed by ul_tx_t2 (+0.1335) and dl_tx_t2 (+0.1593)."\
    "- dl_bitrate is low across all timesteps and contributes positively to anomaly (e.g., +0.0860 at T0, +0.0609 at T1)."\
    "- dl_tx is nonzero (~0.16), close to the normal mean but far below attack peaks — this increases the uplink-downlink imbalance."\
    "- dl_bitrate stays low (~0.0027), much closer to the attack mean (0.0006) than to the normal mean (0.0162)."\
"   - Signal-related features offer context:"\
        "- epre is low (~0.32), closer to the attack mean (0.48) than the normal mean (0.57), suggesting reduced transmit power."\
        "- cqi values reach 1.0, and dl_mcs is high (~0.99), both indicating good radio quality."\
        "- These features had low or negative SHAP impact: epre_t0 (–0.0159), pusch_snr_t1 (–0.0167), cqi_t0 (–0.0059)."\
        "- ul_err and dl_err are zero throughout. LIME shows they slightly decreased anomaly probability (ul_err_t-0: –0.0214, dl_err_t-1: –0.0284), ruling out transient signal issues."\
    "- Global SHAP importance ranks ul_bitrate, ul_tx, and dl_tx among the top contributors overall, meaning this local prediction is in line with the model’s general behavior across the dataset."\
    "Misclassification Likelihood"\
    "Low. The sustained uplink-heavy traffic pattern, combined with elevated retransmissions and suppressed downlink, closely matches known DDoS behavior. Although signal conditions are clean and there are no transmission errors, the dominant traffic characteristics make benign explanations unlikely."\
    "Recommended Mitigation Steps"\
        "- Apply uplink rate limiting or QoS shaping if traffic persists."\
        "- Correlate with flow logs or packet captures to verify session intent and traffic destination."\
        "- If confirmed benign, flag this sample for model feedback and future model fine-tuning."\
        "- If similar patterns are observed across other devices, this may indicate a broader distributed attack in progress."\
]

# Helper to convert JSON to pretty text block (basic for now)
def convert_json_to_string(data):
    return json.dumps(data, indent=2)

# Prepare few-shot messages
messages = [
    {
        "role": "system",
        "content": "You are a cybersecurity expert assisting in interpreting ML model outputs for DDoS detection."
    },
    {
        "role": "user",
        "content": convert_json_to_string(prompt_1)
    },
    {
        "role": "assistant",
        "content": assistant_responses[0]
    },
    {
        "role": "user",
        "content": convert_json_to_string(prompt_2)
    },
    {
        "role": "assistant",
        "content": assistant_responses[1]
    },
    {
        "role": "user",
        "content": convert_json_to_string(query_prompt)
    }
]

# Save to file
output_path = "few_shot_prompt_payload.json"
with open(output_path, "w") as f_out:
    json.dump(messages, f_out, indent=2)

output_path
