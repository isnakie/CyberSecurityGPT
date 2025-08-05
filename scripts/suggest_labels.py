import re
import pandas as pd

# Keyword-based categories
LABEL_KEYWORDS = {
    "Exfiltration - Sensitive Data Exposure": ["gps", "age", "gender", "sexual", "unencrypted", "plaintext", "location"],
    "Exfiltration - Third-Party Data Leak": ["third party", "analytics", "kontagent", "crittercism"],
    "Exploitation - Weak Transport Security": ["http", "sslstrip", "mitm", "invalid certificate", "unencrypted traffic"],
    "Exploitation - Insecure Local Storage": ["unencrypted database", "sqlite", "local storage", "device file"],

    "Credential Access - Token Misuse Risk": ["facebook token", "access token", "auth token"],
    "Discovery - API Exposure via Static Analysis": ["apk", "static analysis", "api endpoint", "decompile"],
    "Exfiltration - Data Overexposure": ["history", "full chat", "last online", "entire chat history"],

    "Discovery - Reconnaissance": ["nmap", "network discovery", "scanned subnet"],
    "Discovery - Enumeration": ["enumerate", "smbclient", "service enumeration"],
    "Exploitation - Code Execution": ["exploit", "remote code execution", "buffer overflow"],
    "Credential Access - Default or Leaked Credentials": ["login page", "admin:admin", "password"],
    "Privilege Escalation - Local/Kernel Exploit": ["root access", "kernel exploit", "sudo"],
    "Persistence - Account Creation": ["add user", "created account", "admin account"],
    "Persistence - Backdoor Deployment": ["meterpreter", "reverse shell", "persistence"],

    "Implementation Flaw - Cryptographic Misuse": ["pbkdf2", "math.random", "constant salt", "hmac", "encryption password"],
    "Implementation Flaw - Insecure Error Handling": ["exception", "crash", "does not handle"],

    "Exploitation - Logic Flaw (TOCTTOU)": ["race condition", "timing"],
    "Exploitation - Trust Boundary Violation": ["wrong domain", "url spoofing"],

    "Exploitation - Unpatched Vulnerability": ["cve", "vulnerable version"],
    "Exploitation - Side Channel Attack": ["charAt", "side-channel"],

    "Support - Policy / Administrative Risk": ["privacy statement", "gdpr", "regulation", "european law"],
    "Support - General Risk Observation": ["mfa", "logging", "lack of", "default"],
    "Background Information": ["project", "introduction", "related work", "method", "material", "legal", "privacy", "authors", "research", "contribution"],
    "Formatting": [],
    
    "Post-Exploitation - Cleanup and Remediation": [
        "house cleaning", "removed all user accounts", "removed passwords",
        "removed meterpreter", "no remnants", "clean up", "delete accounts",
        "remove persistence", "remnants of the penetration test"
    ],
}

NAME_RE = re.compile(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b')
SECTION_HEADER_RE = re.compile(r'^([A-Z]?[0-9]+(\.[0-9]+)*\s+)?[A-Z][a-z].{0,50}$')
PUNCTUATION_LINE_RE = re.compile(r'^[^\w]*$')

def is_formatting_line(line):
    line = line.strip()
    if not line:
        return True
    if len(line.split()) < 5:
        return True
    if PUNCTUATION_LINE_RE.match(line):
        return True
    if SECTION_HEADER_RE.match(line):
        return True
    return False

def suggest_label(paragraph):
    para_lower = paragraph.lower()
    for label, keywords in LABEL_KEYWORDS.items():
        if label != "Background Information":
            if any(keyword in para_lower for keyword in keywords):
                return label
    if (
        re.search(NAME_RE, paragraph)
        or any(keyword in para_lower for keyword in LABEL_KEYWORDS["Background Information"])
    ):
        return "Background Information"
    return "Unknown"

def extract_paragraphs(filename, target_sentences=3):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    paragraphs = []
    current = []
    sentence_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if is_formatting_line(line):
            if len(line) > 2:
                paragraphs.append({"text": line, "label": "Formatting"})
            continue
        current.append(line)
        sentence_count += line.count(".")
        if sentence_count >= target_sentences:
            paragraph = " ".join(current)
            if len(paragraph) > 50:
                paragraphs.append({"text": paragraph.strip(), "label": suggest_label(paragraph)})
            current = []
            sentence_count = 0
    if current:
        paragraph = " ".join(current)
        if len(paragraph) > 50:
            paragraphs.append({"text": paragraph.strip(), "label": suggest_label(paragraph)})
    return pd.DataFrame(paragraphs)

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("input_txt", help="Path to the input .txt report")
    parser.add_argument("output_csv", help="Path to save the labeled CSV")
    args = parser.parse_args()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df = extract_paragraphs(args.input_txt)
    df.to_csv(args.output_csv, index=False)
    print(f"Labeled data saved to {args.output_csv}")

if __name__ == "__main__":
    main()
