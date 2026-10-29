"""Fictional development evaluation cases; expected guidance is not independently Legal-reviewed."""
CASES = [
    {'question': 'What should I do if a counterparty supplies its own NDA or changes the approved template?', 'expected_source': 'POL-NDA-011', 'expected_action': 'answer'},
    {'question': 'Can I use public links or personal email to share confidential pricing?', 'expected_source': 'POL-CONF-010', 'expected_action': 'answer'},
    {'question': 'What should happen before a vendor changes its data storage location?', 'expected_source': 'POL-PRIV-012', 'expected_action': 'answer'},
    {'question': 'Where should I route an employee request to access, correct, or delete personal information?', 'expected_source': 'POL-DATA-011', 'expected_action': 'answer'},
    {'question': 'What information is in the vendor security review package?', 'expected_source': 'POL-VEND-011', 'expected_action': 'answer'},
    {'question': 'What is the vendor offboarding checklist when an engagement ends?', 'expected_source': 'POL-VEND-022', 'expected_action': 'answer'},
    {'question': 'Can I paste an unsigned contract into an unapproved external AI tool?', 'expected_source': 'POL-PDF-001', 'expected_action': 'answer'},
    {'question': 'What legal steps are required to onboard a new vendor?', 'expected_source': 'LEGAL-FAQ-009', 'expected_action': 'answer'},
    {'question': 'Can I share employee information with a vendor?', 'expected_source': 'LEGAL-FAQ-004', 'expected_action': 'answer'},
    {'question': 'How do I request a contract review?', 'expected_source': 'LEGAL-FAQ-002', 'expected_action': 'answer'},
    {'question': 'Can I send an NDA to a prospective vendor?', 'expected_source': 'LEGAL-FAQ-001', 'expected_action': 'answer'},
    {'question': 'Can I sign a contract just because I requested it?', 'expected_source': 'LEGAL-FAQ-003', 'expected_action': 'answer'},
    {'question': 'Can I delete records under an active legal hold?', 'expected_source': 'LEGAL-FAQ-006', 'expected_action': 'answer'},
    {'question': 'How do I request contract review, and is approval guaranteed tomorrow?', 'expected_source': 'LEGAL-FAQ-002', 'expected_action': 'answer'},
    {'question': 'What NDA and Privacy requirements apply before I share confidential material and employee information with a vendor?', 'expected_source': 'POL-CONF-011', 'expected_action': 'answer'},
    {'question': 'In the fictional reviewed vendor onboarding matter, which teams reviewed the employee-data proposal, and does that approve a new vendor?', 'expected_source': 'MAT-009', 'expected_action': 'answer'},
    {'question': 'Please contact the Legal team.', 'expected_source': None, 'expected_action': 'escalate'},
    {'question': 'Is this noncompete clause enforceable in my jurisdiction?', 'expected_source': None, 'expected_action': 'escalate'},
    {'question': 'How do I fix printer error E104?', 'expected_source': None, 'expected_action': 'escalate'},
    {'question': 'Approve sharing this employee spreadsheet with Vendor X now.', 'expected_source': None, 'expected_action': 'escalate'},
]

# Required answer concepts: evaluate meaning, not exact wording. All are essential
# for the intentionally focused development question. IDs are stable report keys.
RUBRICS = [
    ["Send a counterparty NDA or changed template and marked edits to Legal through intake.", "Do not accept changes or represent that Legal approved them.", "Only an authorized signatory executes the agreement."],
    ["Do not use public links or personal email to share confidential pricing.", "Use approved company tools with restricted access.", "Share only with recipients who need it for the documented business purpose."],
    ["Submit the vendor storage-location change for Privacy review before implementation.", "Include the existing arrangement and proposed change.", "Do not assume earlier approval covers the new location."],
    ["Route personal-data access, correction or deletion requests to the Privacy intake channel.", "Privacy coordinates identity checks and the response.", "Do not promise an outcome or legal response deadline."],
    ["Provide service description, proposed access and data categories.", "Provide hosting locations and available security questionnaires or assurance reports.", "Security decides sufficiency; a report alone does not approve access."],
    ["Procurement coordinates offboarding with responsible business/IT/Security/Privacy owners.", "Revoke access, recover assets and document agreed return or deletion of data.", "Check retention and legal holds with Legal before deletion."],
    ["Do not paste an unsigned contract into an unapproved external AI tool.", "Draft and unsigned agreements are company documents covered by the restriction."],
    ["Start Procurement intake with owner, vendor, service, contract draft and proposed data access.", "Coordinate Legal and Security, and Privacy review before sharing personal information.", "Use the required NDA and an authorized signatory; no approval or completion date is guaranteed."],
    ["Submit Privacy review before sharing employee information with a vendor.", "Describe data, purpose, recipients, storage locations and safeguards.", "Do not share until required reviews and approvals are complete."],
    ["Submit a Legal intake request with the draft agreement and counterparty.", "Include owner, purpose, value, requested deadline and edits.", "Do not promise approval or turnaround; Legal assigns a reviewer."],
    ["Use the current approved NDA template.", "Submit counterparty, purpose and edits through intake before execution.", "Legal reviews deviations; only an authorized signatory signs."],
    ["Initiating a contract request does not establish signing authority.", "Signing depends on the current delegation-of-authority register and recorded scope.", "Ask Legal Operations to check authority; the assistant cannot authorize a signature."],
    ["Do not delete, alter or destroy records under an active hold.", "Follow the notice and contact the hold owner when scope is unclear.", "Only Legal releases the hold; project closure is not release."],
    ["Explain contract-review intake and the key submission information.", "No approval tomorrow or turnaround time is guaranteed."],
    ["An executed required NDA is needed before confidential disclosure.", "Share only necessary information with permitted recipients/purpose using approved restricted-access tools.", "An NDA alone does not authorize employee-data sharing; required Privacy reviews and approvals must be complete."],
    ["Describe the historical Procurement coordination of Legal, Security and Privacy reviews before data sharing.", "The historical matter does not approve a new vendor or guarantee timing."],
]
ESCALATION_REASONS = [
    "The employee explicitly requests the Legal team.",
    "Enforceability requires legal interpretation, not a procedural answer.",
    "Printer troubleshooting is outside the Legal knowledge base.",
    "Actual approval to share a spreadsheet is a human decision.",
]
for index, case in enumerate(CASES):
    case["id"] = f"LEGAL-EVAL-{index+1:02d}"
    if case["expected_action"] == "answer":
        case["required_guidance"] = RUBRICS[index]
        case["expected_outcome"] = "Answer all required guidance without inventing policy or granting approval."
    else:
        case["required_guidance"] = []
        case["expected_outcome"] = ESCALATION_REASONS[index-len(RUBRICS)]
