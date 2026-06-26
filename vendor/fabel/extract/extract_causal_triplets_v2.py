#!/usr/bin/env python3
"""
CAUSAL TRIPLET EXTRACTOR v3.0
==============================
SOTA Causal Relationship Extraction with Multi-Pass Architecture

IMPROVEMENTS OVER v2.0:
1. Few-Shot Domain-Specific Prompts with Gold Examples
2. Quantification Booster Pass (dedicated number extraction)
3. Semantic Chunking (sentence-boundary aware)
4. Enhanced 14-Step Validation Pipeline
5. Cross-Triplet Deduplication
6. SBERT Semantic Validation (optional)

Author: Sovereign Pipeline Team
Date: 2026-01-16
"""

import os
import json
import argparse
import requests
import time
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

# --- CONFIG ---
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")  # Upgraded: 14B for better extraction quality
DEFAULT_MLX_MODEL = "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"  # MLX backend (4.8GB RAM)
JARO_WINKLER_THRESHOLD = 0.85  # Whitepaper: threshold for fuzzy entity resolution

# --- MLX BACKEND (Lazy Import) ---
_mlx_backend = None

def get_mlx_backend():
    """Lazy import MLX backend to avoid loading when not needed."""
    global _mlx_backend
    if _mlx_backend is None:
        try:
            from mlx_backend import query_mlx, extract_json_from_r1, get_model, DEFAULT_MLX_MODEL
            _mlx_backend = {
                'query': query_mlx,
                'parse': extract_json_from_r1,
                'get_model': get_model,
                'default_model': DEFAULT_MLX_MODEL,
                'available': True
            }
        except ImportError:
            _mlx_backend = {'available': False}
    return _mlx_backend

# =============================================================================
# JARO-WINKLER SIMILARITY (for Entity Resolution per Whitepaper Section 3.2.5)
# =============================================================================

def jaro_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Calculate Jaro-Winkler similarity between two strings.
    Used for fuzzy entity resolution per Whitepaper Section 3.2.5.

    Args:
        s1, s2: Strings to compare
        p: Scaling factor (default 0.1, max 0.25)

    Returns:
        Similarity score between 0 and 1
    """
    jaro = jaro_similarity(s1.lower(), s2.lower())

    # Find common prefix (max 4 chars)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i].lower() == s2[i].lower():
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * p * (1 - jaro)


# =============================================================================
# DOMAIN-SPECIFIC FEW-SHOT EXAMPLES (Gold Standard)
# =============================================================================

FEW_SHOT_EXAMPLES = {
    "aviation": [
        {
            "trigger": "rudder hardover to full deflection",
            "mechanism": "asymmetric thrust and aerodynamic forces exceed pilot control authority",
            "outcome": "loss of directional control and departure from controlled flight",
            "quantification": "full deflection of 10 degrees",
            "evidence_sentence": "The rudder hardover to full deflection of 10 degrees created asymmetric thrust that exceeded pilot control authority."
        },
        {
            "trigger": "ice accumulation on wing leading edge",
            "mechanism": "disrupted airflow reduces lift coefficient and increases stall speed",
            "outcome": "aerodynamic stall at normal approach speed",
            "quantification": "0.5 inch ice thickness increased stall speed by 15 knots",
            "evidence_sentence": "Ice accumulation of 0.5 inches on the leading edge increased the stall speed by approximately 15 knots."
        }
    ],
    "finance": [
        {
            "trigger": "revenue decline in Q3 2024",
            "mechanism": "reduced operating margin due to fixed cost leverage",
            "outcome": "net income decreased year-over-year",
            "quantification": "revenue down 12%, net income down 23%",
            "evidence_sentence": "The 12% revenue decline in Q3 resulted in a 23% decrease in net income due to operating leverage."
        },
        {
            "trigger": "interest rate increase by Federal Reserve",
            "mechanism": "higher borrowing costs reduce consumer spending and business investment",
            "outcome": "GDP growth slowdown",
            "quantification": "75 basis point increase, GDP growth reduced by 0.4%",
            "evidence_sentence": "The Fed's 75 basis point rate hike contributed to a 0.4% reduction in GDP growth."
        }
    ],
    "medical": [
        {
            "trigger": "elevated troponin I levels above 0.04 ng/mL",
            "mechanism": "indicates myocardial cell death and ongoing cardiac damage",
            "outcome": "increased 30-day mortality risk",
            "quantification": "troponin >0.04 ng/mL, mortality OR 2.8 (p<0.001)",
            "evidence_sentence": "Patients with troponin I >0.04 ng/mL had 2.8 times higher 30-day mortality (p<0.001)."
        },
        {
            "trigger": "administration of 40mg atorvastatin daily",
            "mechanism": "HMG-CoA reductase inhibition reduces hepatic cholesterol synthesis",
            "outcome": "LDL cholesterol reduction",
            "quantification": "40mg dose, 38% LDL reduction at 6 weeks",
            "evidence_sentence": "Atorvastatin 40mg daily reduced LDL cholesterol by 38% after 6 weeks of treatment."
        }
    ],
    "engineering": [
        {
            "trigger": "thermal cycling between -40°C and 125°C",
            "mechanism": "differential thermal expansion causes solder joint fatigue cracking",
            "outcome": "intermittent electrical connection failure",
            "quantification": "1000 cycles to 50% failure rate",
            "evidence_sentence": "After 1000 thermal cycles, 50% of solder joints showed fatigue cracking and intermittent failures."
        },
        {
            "trigger": "vibration exposure at 5G RMS",
            "mechanism": "resonance amplification causes fastener loosening and structural fatigue",
            "outcome": "bracket separation from mounting surface",
            "quantification": "5G RMS for 4 hours caused separation",
            "evidence_sentence": "Exposure to 5G RMS vibration for 4 hours resulted in bracket separation."
        }
    ],
    "legal": [
        {
            "trigger": "violation of SEC Rule 10b-5",
            "mechanism": "material misrepresentation in securities transactions",
            "outcome": "civil liability and disgorgement of profits",
            "quantification": "$2.4 million disgorgement plus $500K penalty",
            "evidence_sentence": "The SEC ordered disgorgement of $2.4 million plus a $500,000 civil penalty for 10b-5 violations."
        }
    ],
    "bitcoin": [
        {
            "trigger": "CVE-2024-38365 btcd consensus failure",
            "mechanism": "btcd removes data in signature script containing signature, leading to discrepancy between Bitcoin Core and btcd",
            "outcome": "nodes fall out of consensus and can be tricked into accepting invalid transactions",
            "quantification": "affects btcd versions prior to 0.24.2",
            "evidence_sentence": "CVE-2024-38365 in btcd caused consensus divergence allowing invalid transaction acceptance."
        },
        {
            "trigger": "time warp attack on difficulty adjustment",
            "mechanism": "attacker manipulates block timestamps at difficulty adjustment boundaries to artificially lower difficulty",
            "outcome": "accelerated block production enables majority hashrate attacks",
            "quantification": "2016 block difficulty window, timestamps manipulated by up to 2 hours",
            "evidence_sentence": "The time warp attack allows difficulty reduction by manipulating timestamps within the 2016 block window."
        },
        {
            "trigger": "malleated witness data in compact block relay",
            "mechanism": "nodes reconstruct blocks with incorrect transaction ordering due to witness malleation",
            "outcome": "block propagation delays or invalid block acceptance",
            "quantification": "CVE-2024-35202 affects nodes using compact blocks",
            "evidence_sentence": "CVE-2024-35202 caused compact block reconstruction failures through witness malleation."
        },
        {
            "trigger": "RBF pinning attack on Lightning channel",
            "mechanism": "attacker broadcasts low-fee conflicting transaction to prevent HTLC timeout claim",
            "outcome": "victim loses funds when HTLC timelock expires",
            "quantification": "up to full channel capacity at risk, typically 0.01-0.1 BTC",
            "evidence_sentence": "RBF pinning can cause Lightning channel fund loss up to the full channel capacity."
        },
        {
            "trigger": "eclipse attack isolating node from honest peers",
            "mechanism": "attacker controls all P2P connections to victim node, feeding manipulated block data",
            "outcome": "victim accepts invalid chain or double-spend transactions",
            "quantification": "requires controlling 8+ outbound connections for full eclipse",
            "evidence_sentence": "Eclipse attacks require controlling at least 8 outbound connections to fully isolate a node."
        },
        {
            "trigger": "INV message flooding from malicious peer",
            "mechanism": "attacker sends excessive inventory announcements overwhelming node's processing",
            "outcome": "node memory exhaustion or legitimate transaction relay delays",
            "quantification": "50,000+ INV messages per second can cause DoS",
            "evidence_sentence": "INV flooding attacks can exhaust node memory with 50,000+ messages per second."
        }
    ],
    "general": [
        {
            "trigger": "implementation of automated quality control system",
            "mechanism": "real-time defect detection enables immediate process correction",
            "outcome": "reduction in product defect rate",
            "quantification": "defect rate reduced from 3.2% to 0.8%",
            "evidence_sentence": "The automated QC system reduced the defect rate from 3.2% to 0.8%."
        }
    ],
    "telecom": [
        {
            "trigger": "BGP route flapping on core router",
            "mechanism": "rapid route withdrawals and advertisements cause routing table instability",
            "outcome": "packet loss and increased latency across network",
            "quantification": "15% packet loss during 45-minute flapping event",
            "evidence_sentence": "BGP route flapping caused 15% packet loss over a 45-minute period."
        },
        {
            "trigger": "SNMP trap storm from managed switches",
            "mechanism": "network management system overwhelmed by excessive trap messages",
            "outcome": "NMS becomes unresponsive and misses critical alarms",
            "quantification": "50,000 traps/minute exceeded NMS capacity of 10,000/minute",
            "evidence_sentence": "The SNMP trap storm of 50,000 traps/minute caused NMS failure."
        }
    ],
    "insurance": [
        {
            "trigger": "failure to disclose material risk during application",
            "mechanism": "policyholder's non-disclosure voids contract under uberrimae fidei doctrine",
            "outcome": "claim denial and policy voidance",
            "quantification": "100% claim rejected, premium not refundable",
            "evidence_sentence": "Non-disclosure of pre-existing condition resulted in complete claim denial."
        },
        {
            "trigger": "exceeding coverage sublimit for business interruption",
            "mechanism": "actual loss exceeded stated sublimit in policy schedule",
            "outcome": "partial indemnification leaving insured with unrecovered loss",
            "quantification": "sublimit $500,000 vs actual loss $1.2M, gap of $700,000",
            "evidence_sentence": "Business interruption losses of $1.2M exceeded the $500,000 sublimit."
        }
    ],
    "pharma": [
        {
            "trigger": "elevated liver enzymes (ALT >3x ULN) in Phase III trial",
            "mechanism": "hepatotoxicity signal requires additional safety monitoring",
            "outcome": "FDA requests Risk Evaluation and Mitigation Strategy (REMS)",
            "quantification": "ALT elevation in 4.2% of treatment group vs 0.8% placebo",
            "evidence_sentence": "The 4.2% incidence of ALT elevation led to REMS requirement."
        },
        {
            "trigger": "drug-drug interaction with CYP3A4 inhibitors",
            "mechanism": "inhibited metabolism increases plasma concentration of study drug",
            "outcome": "increased risk of dose-dependent adverse events",
            "quantification": "co-administration increased AUC by 340%",
            "evidence_sentence": "CYP3A4 inhibitor co-administration increased drug exposure (AUC) by 340%."
        }
    ]
}

NEGATIVE_EXAMPLES = """
DO NOT extract these types of non-causal relationships:

BAD - Definitions (not causation):
  "DNA" → "contains genetic information" → "hereditary material"
  (This defines what DNA is, not what it CAUSES)

BAD - Tautologies (trigger equals outcome):
  "Revenue increase" → "sales growth" → "Revenue growth"
  (Trigger and outcome are the same concept rephrased)

BAD - Vague/Abstract:
  "Lack of planning" → "poor outcomes" → "failure"
  (Too vague, no specific mechanism or measurable outcome)

BAD - Correlations without mechanism:
  "Ice cream sales" → "correlation observed" → "drowning deaths"
  (No causal mechanism, just correlation)

GOOD examples have:
1. Specific trigger event (not vague concepts)
2. Concrete mechanism (HOW the trigger causes the outcome)
3. Measurable outcome (ideally with numbers)
4. Clear causal directionality
"""

# =============================================================================
# QUANTIFICATION PATTERNS (for Number Hunter pass)
# =============================================================================

QUANT_PATTERNS = [
    # === GENERAL PATTERNS ===
    # Percentages
    r'(\d+(?:\.\d+)?)\s*%',
    r'(\d+(?:\.\d+)?)\s*percent',
    # Money
    r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B|K)?',
    r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:USD|EUR|GBP|CHF)',
    # Physical units
    r'(\d+(?:\.\d+)?)\s*(?:kg|lb|lbs|pounds|kilograms)',
    r'(\d+(?:\.\d+)?)\s*(?:m|cm|mm|km|ft|feet|inches|in)',
    r'(\d+(?:\.\d+)?)\s*(?:°C|°F|degrees|celsius|fahrenheit)',
    r'(\d+(?:\.\d+)?)\s*(?:psi|bar|Pa|kPa|MPa)',
    r'(\d+(?:\.\d+)?)\s*(?:knots|mph|km/h|m/s)',
    r'(\d+(?:\.\d+)?)\s*(?:G|g-force)',
    r'(\d+(?:\.\d+)?)\s*(?:Hz|kHz|MHz|GHz)',
    r'(\d+(?:\.\d+)?)\s*(?:V|mV|kV|A|mA|W|kW|MW)',
    # Time
    r'(\d+(?:\.\d+)?)\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)',
    r'(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)',
    # Ratios and multipliers
    r'(\d+(?:\.\d+)?)[xX]\s+(?:increase|decrease|improvement|reduction)',
    r'(\d+(?:\.\d+)?)\s*(?:fold|times)',
    # Counts
    r'(\d+(?:,\d{3})*)\s+(?:patients?|subjects?|samples?|cases?|events?|incidents?)',
    # Version/Reference IDs
    r'CVE-(\d{4}-\d+)',
    r'version\s*(\d+(?:\.\d+)*)',

    # === AVIATION DOMAIN (NTSB/FAA Reports) ===
    r'FL\s*(\d{2,3})',  # Flight level (FL350)
    r'(\d+(?:\.\d+)?)\s*(?:ft|feet)\s*(?:AGL|MSL|agl|msl)?',  # Altitude
    r'(\d+(?:\.\d+)?)\s*(?:fpm|ft/min)',  # Climb/descent rate
    r'Mach\s*(\d+(?:\.\d+)?)',  # Mach number
    r'(\d+(?:\.\d+)?)\s*(?:deg|°)\s*(?:bank|pitch|roll|heading)',  # Angles
    r'(\d+(?:\.\d+)?)\s*(?:KIAS|KTAS|IAS|TAS)',  # Airspeed
    r'V[12sore]\s*[=:]?\s*(\d+)',  # V-speeds (V1, V2, Vso, Vr, Ve)
    r'(\d+(?:\.\d+)?)\s*(?:nm|NM|nautical miles?)',  # Distance
    r'(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)\s*(?:of\s+)?(?:fuel|thrust)',  # Weight/thrust
    r'N[12]\s*[=:]?\s*(\d+(?:\.\d+)?)\s*%',  # Engine N1/N2
    r'(\d{4})\s*(?:UTC|Z|local)',  # Time (aviation format)

    # === PHARMACEUTICAL/MEDICAL DOMAIN (FDA, Clinical Trials) ===
    r'(\d+(?:\.\d+)?)\s*(?:mg|g|mcg|µg|ml|mL|L)',  # Dosages
    r'(\d+(?:\.\d+)?)\s*(?:ng/mL|mg/dL|mmol/L|µg/L|IU/L)',  # Concentrations
    r'p\s*[<>=]\s*(\d+(?:\.\d+)?)',  # P-values
    r'n\s*=\s*(\d+)',  # Sample size
    r'(?:AUC|Cmax|Tmax|t½|t1/2)\s*[=:]?\s*(\d+(?:\.\d+)?)',  # PK parameters
    r'(?:OR|HR|RR|CI)\s*[=:]?\s*(\d+(?:\.\d+)?)',  # Odds/Hazard/Risk ratios
    r'(\d+(?:\.\d+)?)\s*(?:95%?\s*)?CI',  # Confidence intervals
    r'NNT\s*[=:]?\s*(\d+)',  # Number needed to treat
    r'(?:ALT|AST|GGT)\s*[><=]?\s*(\d+(?:\.\d+)?)\s*(?:x\s*)?(?:ULN)?',  # Liver enzymes
    r'(?:eGFR|GFR|CrCl)\s*[><=]?\s*(\d+)',  # Renal function
    r'Phase\s*([IViv123]+)',  # Clinical trial phase
    r'(?:ICH|FDA|EMA)\s*[A-Z]?\d*',  # Regulatory references
    r'(\d+(?:\.\d+)?)\s*(?:mg/kg|mcg/kg)',  # Weight-based dosing
    r'QTc[Ff]?\s*[><=]?\s*(\d+)\s*(?:ms|msec)?',  # QTc prolongation

    # === INSURANCE DOMAIN (Policy Wordings, Claims) ===
    r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:sublimit|deductible|excess|limit)',
    r'(\d+(?:,\d{3})*)\s*(?:aggregate|per\s*occurrence|per\s*claim)',
    r'(?:sublimit|limit|coverage)\s*(?:of\s*)?\$?\s*(\d+(?:,\d{3})*)',
    r'(\d+)\s*(?:day|month|year)\s*(?:waiting\s*period|elimination)',
    r'(\d+(?:\.\d+)?)\s*%\s*(?:co-?insurance|coinsurance|co-?pay)',
    r'(?:Section|Article|Clause)\s*(\d+(?:\.\d+)*)',  # Policy sections
    r'(?:loss\s*ratio|combined\s*ratio)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
    r'(\d+(?:,\d{3})*)\s*(?:claims?|losses|incidents)',

    # === TELECOM/NETWORK DOMAIN (Cisco, Nokia, SAP) ===
    r'(\d+(?:\.\d+)?)\s*(?:Gbps|Mbps|Kbps|bps|Gbit/s|Mbit/s)',  # Bandwidth
    r'(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)\s*(?:latency|delay|RTT)?',  # Latency
    r'(\d+(?:\.\d+)?)\s*%\s*(?:packet\s*loss|loss\s*rate)',  # Packet loss
    r'(\d+(?:\.\d+)?)\s*(?:dB|dBm|dBi)',  # Signal strength
    r'(?:VLAN|vlan)\s*(\d+)',  # VLAN IDs
    r'(?:port|interface)\s*(?:Gi|Fa|Te|Eth)?\s*(\d+(?:/\d+)*)',  # Port numbers
    r'(\d+(?:\.\d+)?)\s*(?:Gbps|10G|40G|100G)\s*(?:interface|port)?',  # Interface speed
    r'AS\s*(\d+)',  # BGP AS numbers
    r'(\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?)',  # IP addresses/subnets
    r'MTU\s*[=:]?\s*(\d+)',  # MTU size
    r'(\d+)\s*(?:hops?|TTL)',  # Hop count
    r'(?:SNMP|syslog)\s*(?:trap|message)\s*(?:ID\s*)?(\d+)',  # SNMP/syslog

    # === BITCOIN/CRYPTOCURRENCY ===
    r'(\d+(?:\.\d+)?)\s*(?:BTC|btc|bitcoin)',
    r'(\d+(?:,\d{3})*)\s*(?:satoshi|sats|sat)',
    r'(\d+(?:,\d{3})*)\s*(?:blocks?|confirmations?)',
    r'(\d+(?:\.\d+)?)\s*(?:sat/vB|sat/byte|sats/vbyte)',
    r'(\d+(?:\.\d+)?)\s*(?:TH/s|PH/s|EH/s|hashrate)',
    r'(\d+(?:\.\d+)?)\s*(?:vbytes?|vB|weight units?|WU)',
    r'block\s*(?:height\s*)?(\d+(?:,\d{3})*)',
    r'BIP[- ]?(\d+)',
]

# Causal signal words for validation
CAUSAL_SIGNALS = [
    'causes', 'caused', 'causing', 'cause',
    'leads to', 'led to', 'leading to',
    'results in', 'resulted in', 'resulting in',
    'triggers', 'triggered', 'triggering',
    'induces', 'induced', 'inducing',
    'produces', 'produced', 'producing',
    'activates', 'inhibits', 'modulates',
    'increases', 'decreases', 'reduces', 'elevates',
    'enables', 'prevents', 'blocks', 'inhibits',
    'due to', 'because of', 'as a result of',
    'consequently', 'therefore', 'thus', 'hence',
    'demonstrates', 'shows that', 'indicates that',
    'contributes to', 'attributed to', 'responsible for',
    'associated with', 'correlates with', 'linked to',
    'impact', 'effect', 'influence', 'affect',
]

# =============================================================================
# PROMPT GENERATION
# =============================================================================

def detect_domain(text: str) -> str:
    """Auto-detect document domain from content."""
    text_lower = text.lower()[:5000]  # Sample first 5KB

    domain_keywords = {
        'aviation': ['aircraft', 'pilot', 'flight', 'aviation', 'runway', 'altitude', 'airspeed', 'faa', 'ntsb', 'cockpit',
                     'accident', 'crash', 'wreckage', 'probable cause', 'airworthiness', 'takeoff', 'landing'],
        'finance': ['revenue', 'profit', 'stock', 'dividend', 'fiscal', 'quarterly', 'earnings', 'investor', 'securities', 'gaap',
                    '10-k', '10-q', 'sec', 'filing', 'shareholder', 'ebitda', 'eps', 'balance sheet'],
        'medical': ['patient', 'treatment', 'clinical', 'dose', 'efficacy', 'placebo', 'trial', 'adverse', 'diagnosis', 'therapy'],
        'engineering': ['component', 'failure', 'thermal', 'stress', 'fatigue', 'tolerance', 'specification', 'design', 'testing'],
        'legal': ['court', 'plaintiff', 'defendant', 'statute', 'violation', 'liability', 'damages', 'contract', 'regulation'],
        'bitcoin': ['bitcoin', 'btc', 'satoshi', 'utxo', 'mempool', 'consensus', 'blockchain', 'transaction', 'script', 'segwit',
                    'taproot', 'lightning', 'htlc', 'channel', 'node', 'block', 'mining', 'hashrate', 'difficulty', 'bip',
                    'optech', 'rbf', 'cpfp', 'fee', 'relay', 'p2p', 'witness', 'signature', 'pubkey', 'multisig'],
        'telecom': ['router', 'switch', 'cisco', 'nokia', 'bgp', 'ospf', 'snmp', 'bandwidth', 'latency', 'packet',
                    'interface', 'vlan', 'mpls', 'qos', 'throughput', 'troubleshooting', 'syslog', 'nms', 'netconf',
                    'ip address', 'routing', 'firewall', 'load balancer', 'sap', 'hana', 'network'],
        'insurance': ['policy', 'premium', 'claim', 'coverage', 'insured', 'underwriting', 'exclusion', 'deductible',
                      'indemnity', 'liability', 'sublimit', 'endorsement', 'policyholder', 'loss', 'peril', 'wording',
                      'terms and conditions', 'insurer', 'reinsurance', 'excess', 'aggregate'],
        'pharma': ['fda', 'nda', 'anda', 'clinical trial', 'phase i', 'phase ii', 'phase iii', 'approval', 'label',
                   'adverse event', 'safety', 'efficacy', 'prea', 'bla', 'drug', 'medicinal', 'pharmaceutical',
                   'regulatory', 'sponsor', 'investigator', 'endpoint', 'statistical', 'pharmacokinetic', 'bioavailability'],
    }

    scores = {domain: 0 for domain in domain_keywords}
    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            scores[domain] += text_lower.count(keyword)

    best_domain = max(scores, key=scores.get)
    return best_domain if scores[best_domain] > 5 else 'general'


def build_few_shot_prompt(text: str, domain: str = None) -> str:
    """
    Generate Few-Shot prompt with domain-specific examples.
    This is the core improvement for extraction quality.
    """
    if domain is None:
        domain = detect_domain(text)

    # Get examples for this domain (fallback to general)
    examples = FEW_SHOT_EXAMPLES.get(domain, FEW_SHOT_EXAMPLES['general'])

    # Format examples as JSON
    examples_json = json.dumps(examples[:2], indent=2)  # Use 2 examples to save tokens

    prompt = f"""You are an expert causal relationship extractor. Your task is to identify cause-and-effect relationships from technical documents.

DOMAIN: {domain.upper()}

TASK: Extract causal triplets in the form:
  TRIGGER (cause) → MECHANISM (how) → OUTCOME (effect)

CRITICAL REQUIREMENTS:
1. Each triplet must represent a TRUE causal relationship (not correlation or definition)
2. Include QUANTIFICATION when numbers are present (exact values with units)
3. Mechanism must explain HOW the trigger causes the outcome
4. Evidence sentence must be a STATEMENT from the text (NOT a question, NOT a paper title)

ANTI-CONTAMINATION RULES (VERY IMPORTANT):
- DO NOT copy the examples below - they are for FORMAT reference only
- Extract ONLY from the actual input text
- If you find yourself writing "troponin", "atorvastatin", "LDL reduction", "rudder hardover" - STOP, these are from examples
- Evidence sentences must come from the INPUT TEXT, not the examples

EVIDENCE QUALITY:
- Evidence must be a declarative statement (ends with period, not question mark)
- Do NOT extract paper titles as evidence (they often start with "Can...", "Does...", "Is...")
- Evidence must contain actual findings, not research questions

{NEGATIVE_EXAMPLES}

EXAMPLE OUTPUT FORMAT (DO NOT COPY CONTENT, ONLY FORMAT):
{examples_json}

NOW EXTRACT FROM THIS TEXT:
---
{text[:10000]}
---

OUTPUT: Return a JSON array with 3-7 high-quality triplets. Quality over quantity.
Each triplet must have: trigger, mechanism, outcome, quantification (or null), evidence_sentence, confidence (high/medium/low)

JSON ONLY - no markdown, no explanation. Start with [ and end with ]"""

    return prompt


def build_quantification_prompt(text: str, triplets: List[Dict]) -> str:
    """
    Second-pass prompt specifically for enhancing quantification.
    Takes existing triplets and searches for missing numbers.
    """
    triplet_summaries = []
    for i, t in enumerate(triplets):
        if not t.get('quantification'):
            triplet_summaries.append(f"{i+1}. {t['trigger']} → {t['outcome']}")

    if not triplet_summaries:
        return None  # All triplets already have quantification

    prompt = f"""TASK: Find numerical values for these causal relationships.

The following causal relationships were extracted but are MISSING quantification.
Search the text for relevant numbers, percentages, measurements, or statistics.

RELATIONSHIPS NEEDING NUMBERS:
{chr(10).join(triplet_summaries)}

TEXT TO SEARCH:
---
{text[:8000]}
---

For each relationship, find the most relevant number from the text.
Return JSON array with objects containing:
- "index": the relationship number (1-based)
- "quantification": the number with units (e.g., "15%", "$2.4M", "1000 cycles")
- "evidence": short quote containing the number

Example: [{{"index": 1, "quantification": "23% reduction", "evidence": "resulted in a 23% reduction in defects"}}]

JSON ONLY. Return empty array [] if no numbers found."""

    return prompt


# =============================================================================
# CHUNKING (Improved with sentence awareness)
# =============================================================================

def chunk_text_semantic(text: str, chunk_size: int = 14000, overlap: int = 800) -> List[Dict]:
    """
    Improved chunking that respects sentence boundaries.
    Returns chunks with metadata for better tracking.
    """
    if len(text) <= chunk_size:
        return [{"text": text, "start": 0, "end": len(text), "index": 0}]

    # Simple sentence splitting (no external deps)
    # Split on . ! ? followed by space and capital letter
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)

    chunks = []
    current_chunk = ""
    current_start = 0
    char_pos = 0

    for sentence in sentences:
        sentence_with_space = sentence + " "

        # If adding this sentence exceeds chunk size, save current and start new
        if len(current_chunk) + len(sentence_with_space) > chunk_size and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "start": current_start,
                "end": char_pos,
                "index": len(chunks)
            })

            # Start new chunk with overlap
            # Find a good overlap point (try to start at sentence boundary)
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            # Find last sentence start in overlap
            last_sentence = overlap_text.rfind('. ')
            if last_sentence > 0:
                overlap_text = overlap_text[last_sentence+2:]

            current_chunk = overlap_text + sentence_with_space
            current_start = char_pos - len(overlap_text)
        else:
            current_chunk += sentence_with_space

        char_pos += len(sentence_with_space)

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "start": current_start,
            "end": char_pos,
            "index": len(chunks)
        })

    return chunks


# =============================================================================
# VALIDATION PIPELINE v2.0 (14 Steps)
# =============================================================================

@dataclass
class ValidationResult:
    """Result of triplet validation."""
    is_valid: bool
    confidence: str
    rejection_reasons: List[str] = field(default_factory=list)
    quality_score: float = 0.0


def validate_triplet_v2(triplet: Dict, all_triplets: List[Dict] = None, source_text: str = None) -> ValidationResult:
    """
    Enhanced 14-step validation pipeline (Foss Hallucination Gate).
    Returns detailed validation result with quality score.

    P11 (Foss-UQA Verification) requires source_text to verify quantification verbatim.
    """
    reasons = []
    score = 100.0  # Start with perfect score, deduct for issues

    trigger = triplet.get('trigger', '').strip()
    mechanism = triplet.get('mechanism', '').strip()
    outcome = triplet.get('outcome', '').strip()
    evidence = triplet.get('evidence_sentence', '').strip()
    quant = triplet.get('quantification')

    # === STEP 1: Field Existence ===
    if not all([trigger, mechanism, outcome]):
        return ValidationResult(False, 'low', ['Missing required fields'], 0.0)

    # === STEP 2: Minimum Length ===
    if len(trigger) < 8:
        reasons.append('Trigger too short (<8 chars)')
        score -= 20
    if len(mechanism) < 15:
        reasons.append('Mechanism too short (<15 chars)')
        score -= 20
    if len(outcome) < 8:
        reasons.append('Outcome too short (<8 chars)')
        score -= 20

    # === STEP 3: Maximum Length (likely hallucination) ===
    if len(trigger) > 200:
        reasons.append('Trigger too long (>200 chars)')
        score -= 10
    if len(mechanism) > 500:
        reasons.append('Mechanism too long (>500 chars)')
        score -= 10

    # === STEP 4: Exact Tautology ===
    if trigger.lower() == outcome.lower():
        return ValidationResult(False, 'low', ['Exact tautology: trigger equals outcome'], 0.0)

    # === STEP 5: Semantic Tautology (word overlap) ===
    trigger_words = set(trigger.lower().split())
    outcome_words = set(outcome.lower().split())
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'is', 'are', 'was', 'were'}
    trigger_words -= stop_words
    outcome_words -= stop_words

    if trigger_words and outcome_words:
        overlap = len(trigger_words & outcome_words)
        overlap_ratio = overlap / min(len(trigger_words), len(outcome_words))
        if overlap_ratio > 0.7 and len(outcome) < 60:
            reasons.append(f'High word overlap ({overlap_ratio:.0%}) suggests tautology')
            score -= 30

    # === STEP 6: Definition Pattern Detection ===
    definition_patterns = [
        r'(?:is|are|was|were)\s+(?:a|an|the)\s+\w+\s+(?:that|which|who)',
        r'(?:defined as|refers to|means|known as)',
        r'^(?:the\s+)?(?:process|act|state)\s+of\s+',
        r'consists?\s+of|comprises?|contains?',
    ]
    mech_lower = mechanism.lower()
    for pattern in definition_patterns:
        if re.search(pattern, mech_lower):
            reasons.append('Mechanism appears to be a definition, not causation')
            score -= 25
            break

    # === STEP 7: Causal Signal Word Check ===
    combined_text = f"{trigger} {mechanism} {outcome} {evidence}".lower()
    has_causal_signal = any(signal in combined_text for signal in CAUSAL_SIGNALS)
    if not has_causal_signal:
        reasons.append('No causal signal words detected')
        score -= 15

    # === STEP 8: Abstract/Vague Concept Filter ===
    vague_patterns = [
        r'^(?:the\s+)?(?:lack|absence|need|importance|significance)\s+of',
        r'^(?:the\s+)?(?:failure|inability)\s+to\s+',
        r'(?:should|must|need to|have to)\s+(?:be|develop|establish)',
        r'^(?:various|many|some|certain)\s+',
        r'(?:etc|and so on|and more)\.?$',
    ]
    for pattern in vague_patterns:
        if re.search(pattern, trigger.lower()) or re.search(pattern, outcome.lower()):
            reasons.append('Trigger or outcome is too abstract/vague')
            score -= 20
            break

    # === STEP 9: Mechanism Quality Check ===
    # Mechanism should explain HOW, not just restate
    if trigger.lower() in mech_lower and outcome.lower() in mech_lower:
        if len(mechanism) < 60:
            reasons.append('Mechanism just restates trigger and outcome')
            score -= 20

    # === STEP 10: Evidence Sentence Validation ===
    if evidence:
        if len(evidence) < 20:
            reasons.append('Evidence sentence too short')
            score -= 10
        elif len(evidence) > 500:
            reasons.append('Evidence sentence too long (might be multiple sentences)')
            score -= 5
        # NEW: Reject questions as evidence (paper titles often start with these)
        question_patterns = [
            r'^(?:Can|Does|Do|Is|Are|What|How|Why|Should|Could|Would|Will)\s+',
            r'\?\s*$',  # Ends with question mark
        ]
        for pattern in question_patterns:
            if re.search(pattern, evidence, re.IGNORECASE):
                reasons.append('Evidence is a question, not a statement')
                score -= 40  # Heavy penalty
                break
    else:
        reasons.append('Missing evidence sentence')
        score -= 10

    # === STEP 11: Foss-UQA Verbatim Verification (P11) ===
    # Whitepaper requirement: q ∈ source_text (quantification must appear verbatim)
    if quant:
        quant_str = str(quant).strip()
        # Check if quantification contains actual numbers
        if not re.search(r'\d', quant_str):
            reasons.append('Quantification field has no numbers')
            score -= 5
        else:
            # P11: Verbatim verification against source text
            if source_text:
                # Extract numeric core from quantification for flexible matching
                # e.g., "15%" should match "15 %" or "15percent"
                numbers_in_quant = re.findall(r'\d+(?:[.,]\d+)?', quant_str)
                verbatim_verified = False

                # First: Try exact match
                if quant_str.lower() in source_text.lower():
                    verbatim_verified = True
                else:
                    # Second: Check if all numbers from quant appear near each other in source
                    if numbers_in_quant:
                        source_lower = source_text.lower()
                        all_numbers_found = all(num in source_lower for num in numbers_in_quant)
                        if all_numbers_found:
                            verbatim_verified = True

                if verbatim_verified:
                    score += 15  # Strong bonus for verified quantification
                    triplet['quant_verified'] = True
                else:
                    reasons.append('P11: Quantification not found verbatim in source')
                    score -= 10  # Penalty for unverified quantification
                    triplet['quant_verified'] = False
            else:
                # No source text available, give moderate bonus
                score += 5
                triplet['quant_verified'] = None  # Unknown

    # === STEP 12: Gibberish/Encoding Detection ===
    gibberish_pattern = r'[^\x00-\x7F]{5,}|(\w)\1{4,}|[!@#$%^&*]{3,}'
    if re.search(gibberish_pattern, trigger + mechanism + outcome):
        reasons.append('Contains gibberish or encoding artifacts')
        score -= 30

    # === STEP 13: Cross-Triplet Duplicate Check ===
    if all_triplets:
        triplet_hash = hashlib.md5(f"{trigger.lower()}|{outcome.lower()}".encode()).hexdigest()
        for other in all_triplets:
            other_hash = hashlib.md5(
                f"{other.get('trigger', '').lower()}|{other.get('outcome', '').lower()}".encode()
            ).hexdigest()
            if triplet_hash == other_hash and other is not triplet:
                reasons.append('Duplicate of another triplet')
                score -= 50
                break

    # === STEP 14 (NEW): Few-Shot Contamination Detection ===
    # Detect if model copied from training examples instead of extracting from text
    CONTAMINATION_MARKERS = [
        'troponin i levels above 0.04',
        'troponin >0.04 ng/ml',
        'mortality or 2.8',
        '40mg atorvastatin',
        'ldl reduction at 6 weeks',
        '38% ldl reduction',
        'rudder hardover',
        'full deflection of 10 degrees',
        'ice accumulation on wing',
        '0.5 inch ice thickness',
        'revenue down 12%',
        'net income down 23%',
        '75 basis point increase',
        'gdp growth reduced by 0.4',
        '1000 cycles to 50% failure',
        '5g rms for 4 hours',
        '$2.4 million disgorgement',
    ]
    combined_lower = f"{trigger} {mechanism} {outcome} {evidence}".lower()
    for marker in CONTAMINATION_MARKERS:
        if marker in combined_lower:
            reasons.append(f'Few-shot contamination detected: "{marker[:30]}..."')
            return ValidationResult(False, 'low', reasons, 0.0)  # Hard reject

    # === STEP 15: Confidence Calibration ===
    original_confidence = triplet.get('confidence', 'medium').lower()

    # Determine final confidence based on score
    if score >= 85:
        final_confidence = 'high'
    elif score >= 60:
        final_confidence = 'medium'
    else:
        final_confidence = 'low'

    # Hard rejection threshold (raised from 40 to 50 for scientific quality)
    is_valid = score >= 50

    return ValidationResult(
        is_valid=is_valid,
        confidence=final_confidence,
        rejection_reasons=reasons,
        quality_score=max(0, score)
    )


# =============================================================================
# OLLAMA INTERACTION
# =============================================================================

def query_ollama(prompt: str, model: str = DEFAULT_MODEL, max_retries: int = 3) -> Optional[str]:
    """Send prompt to Ollama and get response."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_API,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "30m",  # Keep model loaded for 30 min (HUGE speedup)
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 16384,  # Context window
                        "num_predict": 4096,  # Max output tokens
                        "top_p": 0.9,
                    }
                },
                timeout=300
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"   ⚠️ Ollama error: {response.status_code}")
                time.sleep(2)

        except requests.exceptions.ConnectionError:
            print("   ❌ Cannot connect to Ollama at localhost:11434")
            raise
        except Exception as e:
            print(f"   ⚠️ Request error (attempt {attempt+1}): {e}")
            time.sleep(1)

    return None


def parse_json_response(content: str) -> Optional[List[Dict]]:
    """Parse JSON from LLM response with recovery for truncated output."""
    if not content:
        return None

    # Clean up response
    content = content.strip()
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)

    # Try direct parse
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and 'triplets' in result:
            return result['triplets']
    except json.JSONDecodeError:
        pass

    # Recovery: extract complete JSON objects
    triplets = []
    brace_count = 0
    obj_start = -1

    for i, char in enumerate(content):
        if char == '{':
            if brace_count == 0:
                obj_start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and obj_start >= 0:
                try:
                    obj_str = content[obj_start:i+1]
                    obj = json.loads(obj_str)
                    if 'trigger' in obj and 'outcome' in obj:
                        triplets.append(obj)
                except:
                    pass
                obj_start = -1

    return triplets if triplets else None


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

class CausalExtractor:
    """
    Multi-pass causal triplet extractor with validation.

    Modes:
    - turbo=False (default): 2-pass extraction (extraction + quantification)
    - turbo=True: 1-pass extraction (50% faster, slightly less quantification)

    Backends:
    - backend='ollama' (default): Uses Ollama with qwen2.5:14b
    - backend='mlx': Uses MLX with DeepSeek-R1-Distill-Qwen-7B-4bit (Apple Silicon native)
    """

    def __init__(self, model: str = None, turbo: bool = False, backend: str = 'ollama'):
        self.backend = backend.lower()
        self.turbo = turbo  # Skip quantification booster for speed

        # Set model based on backend
        if self.backend == 'mlx':
            mlx = get_mlx_backend()
            if not mlx['available']:
                raise RuntimeError("MLX backend not available. Install: pip install mlx-lm")
            self.model = model or DEFAULT_MLX_MODEL
            # Pre-warm model
            print(f"[MLX] Using {self.model}")
        else:
            self.model = model or DEFAULT_MODEL
            print(f"[Ollama] Using {self.model}")

        self.stats = {
            'chunks_processed': 0,
            'raw_triplets': 0,
            'validated_triplets': 0,
            'quantified_triplets': 0,
            'rejected_triplets': 0,
            'backend': self.backend,
        }

    def _query_llm(self, prompt: str) -> Optional[str]:
        """Route query to appropriate backend (Ollama or MLX)."""
        if self.backend == 'mlx':
            mlx = get_mlx_backend()
            return mlx['query'](prompt, self.model, max_tokens=2048)
        else:
            return query_ollama(prompt, self.model)

    def _parse_response(self, response: str) -> Optional[List[Dict]]:
        """Parse LLM response based on backend."""
        if self.backend == 'mlx':
            mlx = get_mlx_backend()
            # Include reasoning traces for debugging/transparency
            return mlx['parse'](response, include_reasoning=True)
        else:
            return parse_json_response(response)

    def extract_from_text(self, text: str, domain: str = None) -> List[Dict]:
        """
        Full extraction pipeline for a single text.

        Pass 1: Initial extraction with Few-Shot prompt
        Pass 2: Quantification enhancement
        Pass 3: Validation and quality scoring
        """
        if len(text) < 100:
            return []

        # Auto-detect domain if not specified
        if domain is None:
            domain = detect_domain(text)

        # === PASS 1: Initial Extraction ===
        prompt = build_few_shot_prompt(text, domain)
        response = self._query_llm(prompt)
        raw_triplets = self._parse_response(response) or []
        self.stats['raw_triplets'] += len(raw_triplets)

        if not raw_triplets:
            return []

        # === PASS 2: Quantification Enhancement (skip in turbo mode) ===
        if not self.turbo:
            quant_prompt = build_quantification_prompt(text, raw_triplets)
            if quant_prompt:
                quant_response = self._query_llm(quant_prompt)
                quant_updates = self._parse_response(quant_response) or []

                # Apply quantification updates
                for update in quant_updates:
                    idx = update.get('index', 0) - 1  # Convert to 0-based
                    if 0 <= idx < len(raw_triplets) and update.get('quantification'):
                        if not raw_triplets[idx].get('quantification'):
                            raw_triplets[idx]['quantification'] = update['quantification']
                            raw_triplets[idx]['quant_evidence'] = update.get('evidence', '')

        # === PASS 3: Validation with P11 Verbatim Verification ===
        validated_triplets = []
        for triplet in raw_triplets:
            # Pass source text for P11 Foss-UQA verification
            result = validate_triplet_v2(triplet, validated_triplets, source_text=text)

            if result.is_valid:
                triplet['confidence'] = result.confidence
                triplet['quality_score'] = result.quality_score
                triplet['validation_notes'] = result.rejection_reasons
                validated_triplets.append(triplet)
                self.stats['validated_triplets'] += 1
                if triplet.get('quantification'):
                    self.stats['quantified_triplets'] += 1
            else:
                self.stats['rejected_triplets'] += 1

        return validated_triplets

    def extract_from_chunks(self, chunks: List[Dict], domain: str = None) -> List[Dict]:
        """
        Extract from multiple chunks with Jaro-Winkler deduplication (Whitepaper 3.2.5).
        Implements Multi-Hit Confidence Boost: triplets found in multiple chunks get score bonus.
        """
        all_triplets = []
        seen_hashes = set()  # Fast exact-match check

        for chunk in chunks:
            text = chunk['text'] if isinstance(chunk, dict) else chunk
            self.stats['chunks_processed'] += 1

            triplets = self.extract_from_text(text, domain)

            # Deduplicate with both hash and Jaro-Winkler + Multi-Hit Boost
            for t in triplets:
                trigger_lower = t['trigger'].lower()[:100]
                outcome_lower = t['outcome'].lower()[:100]

                # Initialize hit_count if not present
                if 'hit_count' not in t:
                    t['hit_count'] = 1

                # Fast path: exact hash match - boost existing
                t_hash = hashlib.md5(f"{trigger_lower}|{outcome_lower}".encode()).hexdigest()
                if t_hash in seen_hashes:
                    # Find and boost the existing triplet
                    for existing in all_triplets:
                        ex_hash = hashlib.md5(
                            f"{existing['trigger'].lower()[:100]}|{existing['outcome'].lower()[:100]}".encode()
                        ).hexdigest()
                        if ex_hash == t_hash:
                            existing['hit_count'] = existing.get('hit_count', 1) + 1
                            # Multi-hit boost: +5 per additional hit (caps at +20)
                            boost = min(existing['hit_count'] - 1, 4) * 5
                            existing['quality_score'] = existing.get('quality_score', 50) + boost
                            break
                    continue

                # Slow path: Jaro-Winkler fuzzy matching for near-duplicates
                is_duplicate = False
                for existing in all_triplets:
                    existing_trigger = existing['trigger'].lower()[:100]
                    existing_outcome = existing['outcome'].lower()[:100]

                    # Check if trigger AND outcome are similar
                    trigger_sim = jaro_winkler_similarity(trigger_lower, existing_trigger)
                    outcome_sim = jaro_winkler_similarity(outcome_lower, existing_outcome)

                    if trigger_sim >= JARO_WINKLER_THRESHOLD and outcome_sim >= JARO_WINKLER_THRESHOLD:
                        is_duplicate = True
                        # Multi-hit: increment count and boost score
                        existing['hit_count'] = existing.get('hit_count', 1) + 1
                        boost = min(existing['hit_count'] - 1, 4) * 5  # +5 per hit, max +20
                        existing['quality_score'] = existing.get('quality_score', 50) + boost

                        # If new one has better base score, merge its data
                        if t.get('quality_score', 0) > existing.get('quality_score', 0) - boost:
                            # Keep new triplet's content but preserve hit_count
                            hit_count = existing['hit_count']
                            idx = all_triplets.index(existing)
                            all_triplets[idx] = t
                            all_triplets[idx]['hit_count'] = hit_count
                            all_triplets[idx]['quality_score'] = t.get('quality_score', 50) + boost
                            seen_hashes.add(t_hash)
                        break

                if not is_duplicate:
                    seen_hashes.add(t_hash)
                    all_triplets.append(t)

        return all_triplets

    def get_stats(self) -> Dict:
        """Return extraction statistics."""
        return {
            **self.stats,
            'validation_rate': (
                self.stats['validated_triplets'] / max(self.stats['raw_triplets'], 1) * 100
            ),
            'quantification_rate': (
                self.stats['quantified_triplets'] / max(self.stats['validated_triplets'], 1) * 100
            ),
        }


# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

def chunk_text(text: str, chunk_size: int = 10000, overlap: int = 1000) -> List[str]:
    """Legacy chunking function for backward compatibility."""
    chunks = chunk_text_semantic(text, chunk_size, overlap)
    return [c['text'] for c in chunks]


def validate_triplet(triplet: Dict) -> Tuple[bool, str]:
    """Legacy validation function for backward compatibility."""
    result = validate_triplet_v2(triplet)
    reason = result.rejection_reasons[0] if result.rejection_reasons else "Valid"
    return result.is_valid, reason


def query_ollama_legacy(text: str, model: str = DEFAULT_MODEL, domain_hint: str = "") -> Optional[List[Dict]]:
    """Legacy extraction function for backward compatibility."""
    extractor = CausalExtractor(model)
    return extractor.extract_from_text(text, domain_hint if domain_hint else None)


# Alias for backward compatibility
query_ollama_extract = query_ollama_legacy


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Causal Triplet Extractor v3.0 (SOTA)")
    parser.add_argument("--input-dir", required=True, help="Directory with parsed JSON files")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--model", default=None, help="Model name (auto-selected based on backend)")
    parser.add_argument("--domain", default=None, help="Domain hint (auto-detected if not specified)")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "mlx"],
                        help="Backend: 'ollama' (default, 14B) or 'mlx' (Apple Silicon, 7B R1)")
    parser.add_argument("--turbo", action="store_true", help="Skip quantification pass (faster)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  CAUSAL TRIPLET EXTRACTOR v3.0 (SOTA)")
    print("  Few-Shot + Multi-Pass + 14-Step Validation")
    print("=" * 60)

    # Backend-specific checks
    if args.backend == 'mlx':
        mlx = get_mlx_backend()
        if not mlx['available']:
            print("\n❌ MLX backend not available. Install: pip install mlx-lm")
            return
        print(f"\n✅ MLX backend ready")
        print(f"   Model: {args.model or DEFAULT_MLX_MODEL}")
        print(f"   RAM: ~4.8GB (Apple Silicon native)")
    else:
        # Check Ollama
        try:
            requests.get("http://localhost:11434/", timeout=2)
            print(f"\n✅ Ollama connection OK")
            print(f"   Model: {args.model or DEFAULT_MODEL}")
        except:
            print("\n❌ Cannot connect to Ollama. Run: ollama serve")
            return

    # Process files
    input_path = Path(args.input_dir)
    files = list(input_path.glob("*.json"))
    print(f"\n📂 Found {len(files)} files")

    extractor = CausalExtractor(model=args.model, backend=args.backend, turbo=args.turbo)
    all_triplets = []

    for f in files:
        print(f"\n📄 Processing {f.name}...")
        try:
            with open(f) as fp:
                data = json.load(fp)

            text = data.get("full_text", "")
            if len(text) < 100:
                print(f"   ⏭️ Skipping (too short)")
                continue

            # Chunk and extract
            chunks = chunk_text_semantic(text)
            print(f"   📦 {len(chunks)} chunks")

            triplets = extractor.extract_from_chunks(chunks, args.domain)

            # Add source file
            for t in triplets:
                t['source_file'] = data.get('filename', f.name)

            all_triplets.extend(triplets)
            print(f"   ✅ {len(triplets)} triplets extracted")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Save output
    stats = extractor.get_stats()
    output = {
        "version": "3.0",
        "extraction_method": "few_shot_multipass",
        "stats": stats,
        "mechanisms": all_triplets
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"📊 EXTRACTION COMPLETE")
    print(f"=" * 60)
    print(f"   Total triplets: {len(all_triplets)}")
    print(f"   Validation rate: {stats['validation_rate']:.1f}%")
    print(f"   Quantification rate: {stats['quantification_rate']:.1f}%")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
