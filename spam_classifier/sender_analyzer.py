
import ipaddress
import re
from email.utils import parseaddr


FREE_EMAIL_PROVIDERS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "proton.me",
    "protonmail.com",
}

ORGANIZATION_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited",
    "corp", "corporation", "company", "co", "official",
}


def normalize_tokens(value):
    """Return useful words for comparing organization and domain names."""
    words = re.findall(r"[a-z0-9]+", (value or "").lower())
    return {
        word for word in words
        if word not in ORGANIZATION_SUFFIXES and len(word) > 1
    }


def extract_address(value):
    """Extract an email address from plain text or a display-name format."""
    _, address = parseaddr((value or "").strip())
    return address.strip()


def validate_email(address):
    """Basic email format validation; this does not verify mailbox ownership."""
    if not address or len(address) > 254 or address.count("@") != 1:
        return False

    local, domain = address.rsplit("@", 1)

    if not local or len(local) > 64 or not domain:
        return False

    if local.startswith(".") or local.endswith(".") or ".." in local:
        return False

    if not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+", local):
        return False

    if domain.startswith("[") and domain.endswith("]"):
        try:
            ipaddress.ip_address(domain[1:-1])
            return True
        except ValueError:
            return False

    try:
        ascii_domain = domain.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if len(ascii_domain) > 253:
        return False

    labels = ascii_domain.split(".")
    if len(labels) < 2:
        return False

    for label in labels:
        if (
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or not re.fullmatch(r"[A-Za-z0-9-]+", label)
        ):
            return False

    return bool(re.fullmatch(r"[A-Za-z]{2,63}", labels[-1]))


def get_domain(address):
    if "@" not in address:
        return ""
    return address.rsplit("@", 1)[1].lower().rstrip(".")


def analyze_sender(sender_email, display_name="", organization="", reply_to=""):
    """Analyze user-entered sender details using transparent heuristics."""
    sender_email = extract_address(sender_email)
    reply_to = extract_address(reply_to) if reply_to.strip() else ""

    findings = []
    score = 0

    valid_sender = validate_email(sender_email)
    domain = get_domain(sender_email) if valid_sender else ""

    if not valid_sender:
        score += 40
        findings.append({
            "level": "warning",
            "title": "Invalid sender address format",
            "detail": "The sender address does not match the expected email format."
        })
    else:
        findings.append({
            "level": "info",
            "title": "Sender address format is valid",
            "detail": "The address has a plausible format. This does not confirm that the mailbox exists."
        })

        try:
            ipaddress.ip_address(domain.strip("[]"))
            is_ip_domain = True
        except ValueError:
            is_ip_domain = False

        if is_ip_domain:
            score += 25
            findings.append({
                "level": "warning",
                "title": "IP address used as the domain",
                "detail": "A domain written as an IP address is unusual for an ordinary business sender."
            })

        if domain and any(
            label.startswith("xn--") for label in domain.split(".")
        ):
            score += 20
            findings.append({
                "level": "warning",
                "title": "Internationalized domain detected",
                "detail": "The domain contains punycode. Check it carefully for lookalike characters."
            })

        if domain in FREE_EMAIL_PROVIDERS:
            findings.append({
                "level": "info",
                "title": "Public email provider",
                "detail": "This address uses a public email service. That is common and is not evidence of fraud by itself."
            })

        if domain and domain.count(".") >= 4:
            score += 10
            findings.append({
                "level": "caution",
                "title": "Unusually deep subdomain",
                "detail": "The sender uses several domain levels. Check that the registered domain belongs to the expected organization."
            })

    display_name = (display_name or "").strip()
    organization = (organization or "").strip()

    if display_name and organization:
        name_tokens = normalize_tokens(display_name)
        org_tokens = normalize_tokens(organization)

        if org_tokens and not name_tokens.intersection(org_tokens):
            score += 10
            findings.append({
                "level": "caution",
                "title": "Display name differs from claimed organization",
                "detail": "The display name does not contain recognizable words from the organization you entered. This can have legitimate explanations."
            })

    if valid_sender and organization:
        org_tokens = normalize_tokens(organization)
        domain_labels = {
            token for token in re.findall(r"[a-z0-9]+", domain)
            if len(token) > 1
        }

        if org_tokens and not org_tokens.intersection(domain_labels):
            score += 20
            findings.append({
                "level": "warning",
                "title": "Domain does not obviously match the organization",
                "detail": "The domain does not contain recognizable organization words. Some legitimate organizations use different domain names, so verify it independently."
            })

    if reply_to:
        valid_reply = validate_email(reply_to)

        if not valid_reply:
            score += 20
            findings.append({
                "level": "warning",
                "title": "Invalid reply-to address",
                "detail": "The reply-to address does not match the expected email format."
            })
        else:
            reply_domain = get_domain(reply_to)

            if valid_sender and reply_domain != domain:
                score += 25
                findings.append({
                    "level": "warning",
                    "title": "Sender and reply-to domains differ",
                    "detail": "Replies would go to a different domain. This can be legitimate, but should be checked when the message claims to be from an organization."
                })
            else:
                findings.append({
                    "level": "info",
                    "title": "Reply-to domain matches",
                    "detail": "The reply-to domain matches the sender domain."
                })

    score = min(score, 100)

    if score >= 50:
        risk_level = "High"
        risk_description = "Several warning indicators were detected. Verify the sender through an independently obtained contact method."
    elif score >= 25:
        risk_level = "Medium"
        risk_description = "Some warning indicators were detected. Check the sender and domain before trusting the message."
    else:
        risk_level = "Low"
        risk_description = "Few warning indicators were detected in the fields supplied. This does not prove the sender is safe."

    if not findings:
        findings.append({
            "level": "info",
            "title": "No findings",
            "detail": "Enter sender details to receive a report."
        })

    return {
        "sender_email": sender_email,
        "domain": domain or "Unable to determine",
        "display_name": display_name or "Not provided",
        "organization": organization or "Not provided",
        "reply_to": reply_to or "Not provided",
        "score": score,
        "risk_level": risk_level,
        "risk_description": risk_description,
        "findings": findings,
    }
