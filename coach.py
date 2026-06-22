import ollama


def build_session_summary(swings):
    if not swings:
        return None
    
    n = len(swings)
    club = swings[0].get("club", "unknown")
    
    avg_tempo = sum(s["tempo_ratio"] for s in swings) / n
    avg_head = sum(s["head_movement"] for s in swings) / n
    avg_hand_height = sum(s["top_hand_height"] for s in swings) / n
    avg_spine = sum(s["spine_maintenance"] for s in swings) / n
    
    tempo_values = [s["tempo_ratio"] for s in swings]
    tempo_deviation = round(max(tempo_values) - min(tempo_values), 2)
    
    return {
        "swings_analyzed": n,
        "club": club,
        "avg_tempo": round(avg_tempo, 2),
        "avg_head_movement": round(avg_head, 3),
        "avg_top_hand_height": round(avg_hand_height, 1),
        "avg_spine_maintenance": round(avg_spine, 1),
        "tempo_consistency": tempo_deviation
    }


def analyze_session(swings):
    summary = build_session_summary(swings)
    
    if summary is None:
        return "No swing data available to analyze."
    
    prompt = f"""You are a PGA-certified golf instructor analyzing a student's practice session.

SESSION DATA:
- Swings analyzed: {summary['swings_analyzed']}
- Club: {summary['club']}
- Average tempo ratio: {summary['avg_tempo']}:1 (tour target: 3.0:1)
- Tempo consistency (range): {summary['tempo_consistency']} (lower is better, <0.5 is excellent)
- Average head movement: {summary['avg_head_movement']} (target: <0.03, <0.05 is acceptable)
- Average top hand height: {summary['avg_top_hand_height']}% of torso length
- Average spine maintenance deviation: {summary['avg_spine_maintenance']}° (target: <5°)

Respond in EXACTLY this format, no extra text:

PRIMARY ISSUE: [one sentence identifying the single biggest problem]

WHY IT MATTERS: [one sentence explaining the impact on ball striking]

DRILL: [name of a specific well-known golf drill]
Setup: [one sentence]
Execution: [two sentences max]
Reps: [number and sets]

POSITIVE: [one thing they are doing well based on the data]

Be specific and actionable. No generic advice."""

    response = ollama.chat(
        model='llama3.1',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        options={'temperature': 0.3}
    )
    
    return response['message']['content']