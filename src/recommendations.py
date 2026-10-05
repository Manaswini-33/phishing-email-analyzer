"""
src/recommendations.py
Actionable Cybersecurity Guidance, Incident Response Playbooks & Security Disclaimers.
"""

from typing import List, Dict, Any


def generate_security_recommendations(
    risk_level: str,
    phishing_type: str,
    heuristic_penalties: Dict[str, float],
    is_anomaly: bool = False
) -> Dict[str, Any]:
    """
    Generate tailored, actionable cybersecurity response guidance based on
    calibrated risk level, predicted phishing threat vector, and behavioral indicators.

    Args:
        risk_level: 'Low', 'Medium', 'High', or 'Critical'.
        phishing_type: Detected category (e.g., 'Credential Harvesting', 'CEO Impersonation').
        heuristic_penalties: Dictionary of active heuristic flags.
        is_anomaly: Whether the email was flagged as an unsupervised anomaly.

    Returns:
        Dict containing:
            - 'primary_action': High-level immediate verdict/directive
            - 'action_steps': List[str] step-by-step containment instructions
            - 'technical_precautions': List[str] header & domain verification steps
            - 'disclaimer': Official cybersecurity legal & operational disclaimer
    """
    action_steps: List[str] = []
    technical_precautions: List[str] = []

    # 1. Level-specific Immediate Actions
    if risk_level == "Critical":
        primary_action = "CRITICAL THREAT: ISOLATE & DO NOT INTERACT"
        action_steps.extend([
            "DO NOT click any hyperlinks, buttons, or downloadable attachments.",
            "DO NOT reply to the sender or attempt to negotiate/inquire via email.",
            "If you entered your password or MFA code, immediately change your corporate credentials and notify your Security Operations Center (SOC).",
            "Forward the email as an RFC 822 (.eml) attachment to your security team (e.g., security@company.com) for automated sandbox quarantine."
        ])
    elif risk_level == "High":
        primary_action = "HIGH RISK: SUSPICIOUS COMMUNICATION DETECTED"
        action_steps.extend([
            "Treat all embedded links and call-to-action buttons with extreme suspicion.",
            "Verify the sender's identity through a trusted out-of-band communication channel (e.g., direct Slack message, verified internal directory phone number).",
            "Do not disclose confidential company data, financial records, or credentials.",
            "Report the email using your organization's 'Report Phishing' button."
        ])
    elif risk_level == "Medium":
        primary_action = "MODERATE RISK: EXERCISE CAUTION & VERIFY SENDER"
        action_steps.extend([
            "Inspect the exact sender address domain (look for lookalike typosquatting domains like @micros0ft.com or @paypal-support.info).",
            "Hover over links (without clicking) to inspect the true destination URL in the status bar.",
            "If this claims to be an unexpected invoice or shipping notification, verify directly on the official vendor portal."
        ])
    else:
        primary_action = "LOW RISK: ROUTINE COMMUNICATION PATTERN"
        action_steps.extend([
            "Email appears consistent with standard legitimate communication patterns.",
            "Standard cybersecurity hygiene still applies: never share unencrypted passwords via email.",
            "Ensure Multi-Factor Authentication (MFA) remains active on your account."
        ])

    # 2. Threat-Type Specific Guidance
    p_type_lower = phishing_type.lower()
    if "credential" in p_type_lower or "password" in p_type_lower:
        technical_precautions.append(
            "Credential Harvesting Threat: Legitimate IT departments will never ask for your password or OTP over an unauthenticated web link."
        )
    if "invoice" in p_type_lower or "financial" in p_type_lower or "wire" in p_type_lower:
        technical_precautions.append(
            "Financial Wire Protocol: Any modification to bank accounts or payment instructions requires dual-custody voice authorization."
        )
    if "ceo" in p_type_lower or "executive" in p_type_lower:
        technical_precautions.append(
            "Executive Impersonation (Business Email Compromise): Attackers often leverage urgency and fake mobile signatures to bypass procurement controls."
        )
    if "delivery" in p_type_lower or "package" in p_type_lower:
        technical_precautions.append(
            "Postal/Delivery Scam: Track packages only by entering the tracking number manually on official carrier websites (USPS, FedEx, DHL, UPS)."
        )

    # 3. Anomaly Flagging Guidance
    if is_anomaly:
        technical_precautions.append(
            "Zero-Day Anomaly Alert: This message exhibits novel structural characteristics distinct from standard email baselines. Additional manual analyst review is advised."
        )

    # 4. Standard Technical Precautions
    technical_precautions.extend([
        "Check email authentication headers: SPF (Sender Policy Framework), DKIM (DomainKeys Identified Mail), and DMARC alignment status.",
        "Check if the Display Name matches the actual SMTP envelope `Return-Path` address."
    ])

    disclaimer = (
        "SECURITY DISCLAIMER: This AI-powered risk assessment is generated via statistical and machine learning "
        "pattern recognition models for informational and threat triage purposes. It does not replace enterprise perimeter "
        "firewalls, email gateway filters, Endpoint Detection and Response (EDR) solutions, or human security analyst "
        "judgment. No cybersecurity system guarantees 100% detection of zero-day social engineering vectors."
    )

    return {
        "primary_action": primary_action,
        "action_steps": action_steps,
        "technical_precautions": technical_precautions,
        "disclaimer": disclaimer
    }
