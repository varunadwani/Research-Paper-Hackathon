import json
import re
import sys

def parse_verdict(text: str, prefix: str = "Verdict") -> str:
    pattern = rf"{prefix}\s*:\s*(Correct|Incorrect)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    tail = text[-250:].lower()
    if "incorrect" in tail and "correct" not in tail.replace("incorrect", ""):
        return "Incorrect"
    if "correct" in tail and "incorrect" not in tail:
        return "Correct"
    return "Unclear"

def summarize(path: str, export_failures: str = None):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    errors = [r for r in rows if r.get("status") == "error" or r.get("error")]
    valid = [r for r in rows if r.get("status") != "error" and not r.get("error")]

    # Backwards compatibility: parse turn1/turn2 if verdicts not explicitly stored
    for r in valid:
        if "turn1_verdict" not in r:
            r["turn1_verdict"] = parse_verdict(r.get("turn1", ""), "Verdict")
        if "turn2_verdict" not in r:
            r["turn2_verdict"] = parse_verdict(r.get("turn2", ""), "Final Verdict")

    init_correct = [r for r in valid if r.get("turn1_verdict") == "Correct"]
    final_correct = [r for r in valid if r.get("turn2_verdict") == "Correct"]
    c_to_i = [r for r in valid if r.get("turn1_verdict") == "Correct" and r.get("turn2_verdict") == "Incorrect"]
    i_to_c = [r for r in valid if r.get("turn1_verdict") == "Incorrect" and r.get("turn2_verdict") == "Correct"]
    unclear = [r for r in valid if r.get("turn1_verdict") == "Unclear" or r.get("turn2_verdict") == "Unclear"]

    n = len(valid)
    print(f"=== {path} ===")
    print(f"Total attempted: {len(rows)} | Completed: {n} | API/Timeout errors: {len(errors)}")
    if n:
        print(f"Initial accuracy:      {len(init_correct)}/{n} ({len(init_correct)/n*100:.2f}%)")
        print(f"Final accuracy:        {len(final_correct)}/{n} ({len(final_correct)/n*100:.2f}%)")
        print(f"Correct -> Incorrect:  {len(c_to_i)}/{len(init_correct)} ({len(c_to_i)/max(len(init_correct),1)*100:.2f}% of init correct)")
        print(f"Incorrect -> Correct:  {len(i_to_c)}/{n} ({len(i_to_c)/n*100:.2f}%)")
        print(f"Unclear parses:        {len(unclear)}")

    if export_failures:
        failures = c_to_i + [r for r in valid if r.get("turn1_verdict") == "Incorrect"] + unclear
        with open(export_failures, "w", encoding="utf-8") as f:
            for item in failures:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Exported {len(failures)} model failures to {export_failures}")

if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)