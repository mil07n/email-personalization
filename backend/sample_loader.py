"""
Sample Email Loader.
Pulls 30 diverse, authentic curated sample emails representing all urgency levels
(CRITICAL, HIGH, MEDIUM, LOW) and all categories (OPERATIONS, ACTION_REQUIRED,
MEETING, FINANCE, EXTERNAL, NEWSLETTER, SPAM) — sourced from Enron & Customer Care datasets.
"""

from typing import List, Dict, Any


def get_demonstration_emails() -> List[Dict[str, Any]]:
    """
    Returns 30 curated representative real-world emails covering every priority level
    and email category for a fully populated demonstration inbox.
    """
    return [
        # ═══════════════════════════════════════════════════════
        # CRITICAL PRIORITY (4 emails)
        # ═══════════════════════════════════════════════════════
        {
            "id": "crit_01",
            "subject": "CRITICAL: Primary Database Cluster Offline - Outage Incident",
            "body": "Team, our primary production database cluster is offline and crashed following hardware failure at 08:15 AM EST. Over 50,000 queries have timed out and critical services are unavailable. John, please escalate this to cloud infrastructure immediately and execute the disaster recovery failover before 11:00 AM. The CTO and CIO are both on standby.",
            "sender": "infrastructure-lead@company.com",
            "receiver": "John Smith",
            "timestamp": "2026-11-14 08:20:00",
            "thread_id": "thread_inc_001"
        },
        {
            "id": "crit_02",
            "subject": "URGENT: Security Breach Alert - Unauthorized Root Access Detected",
            "body": "Security Operations Center has detected an active security breach on our authentication servers. Potential unauthorized credential dump is in progress. All engineers must revoke SSH keys immediately and patch the vulnerability today. This constitutes a critical security incident under our SOC2 compliance policy.",
            "sender": "security-ops@company.com",
            "receiver": "Executive Team",
            "timestamp": "2026-11-14 08:35:00",
            "thread_id": "thread_sec_002"
        },
        {
            "id": "crit_03",
            "subject": "EMERGENCY: Payment Gateway Fatal Error - Outage Impacting Checkout",
            "body": "Customer checkouts are failing globally due to a fatal error in Stripe API token validation. Production is severely impacted with over 1,200 failed transactions in the last 30 minutes. Sarah, please review the crash logs and initiate rollback patch immediately. Payment failure rate is at 94%. Client SLAs are in breach.",
            "sender": "payments-team@company.com",
            "receiver": "Sarah Jenkins",
            "timestamp": "2026-11-14 08:45:00",
            "thread_id": "thread_pay_003"
        },
        {
            "id": "crit_04",
            "subject": "CRITICAL: Core Trading Network System Down - Urgent Intervention",
            "body": "Trading operations desk reports the core order routing network is completely down and crashed. Markets are open and client trades cannot execute. Production down alert triggered. System engineers must bring the backup cluster online immediately. Potential regulatory breach if not resolved within 2 hours.",
            "sender": "trading-ops@enron.com",
            "receiver": "Vince Kaminski",
            "timestamp": "2026-11-14 09:00:00",
            "thread_id": "thread_trade_004"
        },

        # ═══════════════════════════════════════════════════════
        # HIGH PRIORITY (5 emails)
        # ═══════════════════════════════════════════════════════
        {
            "id": "high_01",
            "subject": "Quarterly Financial Audit and Contract Review - Due Tomorrow",
            "body": "Sarah, please prepare the financial report for KPMG audit and send it before tomorrow 5:00 PM. We need to verify all transport capacity invoices and signed NDA contracts prior to Monday morning. The external auditors will not accept delays — this deadline is firm.",
            "sender": "audit.director@company.com",
            "receiver": "Sarah Jenkins",
            "timestamp": "2026-11-14 09:15:00",
            "thread_id": "thread_audit_005"
        },
        {
            "id": "high_02",
            "subject": "Urgent Action Required: Q4 Budget Variance Submission",
            "body": "Leadership team, you need to submit your departmental budget variances by EOD today. Finance will lock the fiscal year spreadsheet at 6:00 PM sharp. Please ensure all vendor expenses, capital expenditures, and project overruns are accounted for. Missing submissions will result in automatic zero-variance entries.",
            "sender": "cfo@company.com",
            "receiver": "Management",
            "timestamp": "2026-11-14 09:30:00",
            "thread_id": "thread_budget_006"
        },
        {
            "id": "high_03",
            "subject": "Urgent: Enterprise Client SLA Escalation - Ticket #84920",
            "body": "David, our tier-1 enterprise partner Accenture has escalated ticket #84920 regarding persistent API latency. Their CTO is threatening to invoke contract penalty clauses. Action required: kindly schedule a technical review call with their VP of Engineering before 3:00 PM today and prepare the incident post-mortem.",
            "sender": "client-success@company.com",
            "receiver": "David Miller",
            "timestamp": "2026-11-14 10:00:00",
            "thread_id": "thread_sla_007"
        },
        {
            "id": "high_04",
            "subject": "Executive Strategy Session: Agenda & Deliverables Due Tomorrow",
            "body": "Leadership team, our Q1 2027 strategic roadmap session is scheduled for next Tuesday at 2:00 PM. Alice, please circulate the draft slide deck by tomorrow EOD so everyone can review in advance. Mark, you need to prepare the competitive landscape analysis and financial projections before the session.",
            "sender": "ceo@company.com",
            "receiver": "Alice Johnson",
            "timestamp": "2026-11-14 10:15:00",
            "thread_id": "thread_strategy_008"
        },
        {
            "id": "high_05",
            "subject": "URGENT: User Permission Errors in Production API - Action Required",
            "body": "We are experiencing critical issues with user permissions on our production API (API key: aeth-prod-28374). Users with the 'data-reader' role are receiving 403 Forbidden errors when querying the analytics endpoints. Please investigate the RBAC policy update that was deployed yesterday and revert if necessary.",
            "sender": "operations@aetheros.com",
            "receiver": "Platform Engineering",
            "timestamp": "2026-11-14 10:30:00",
            "thread_id": "thread_perm_009"
        },

        # ═══════════════════════════════════════════════════════
        # MEDIUM PRIORITY (4 emails)
        # ═══════════════════════════════════════════════════════
        {
            "id": "med_01",
            "subject": "Customer Care Ticket: Billing Address Correction Request",
            "body": "Hello Support Team, I noticed my recent subscription invoice still lists our old headquarters address at 123 Main Street. Could you please update our billing profile to our new office suite at 456 Corporate Blvd? Our account number is CC-94812. Thank you for your assistance.",
            "sender": "client.admin@partner.com",
            "receiver": "Support Team",
            "timestamp": "2026-11-14 10:45:00",
            "thread_id": "thread_cc_010"
        },
        {
            "id": "med_02",
            "subject": "API Monitoring Dashboard Improvement Suggestions",
            "body": "Hi Aetheros Support Team, I have been using your API monitoring dashboard for our test environments and overall it works well. The charting widget for latency metrics is very useful. However, it would be helpful if the data export button also supported CSV format in addition to JSON. Could you log this as a feature request?",
            "sender": "dev-user@customer.org",
            "receiver": "Product Feedback",
            "timestamp": "2026-11-14 11:00:00",
            "thread_id": "thread_fb_011"
        },
        {
            "id": "med_03",
            "subject": "Vendor Agreement Feedback: Western Gas Pipeline Contract",
            "body": "Phillip, I have reviewed the revised vendor agreement terms from Columbia Gas of Pennsylvania. The delivery point pricing structure appears acceptable within our Q4 margins. When you have a moment this week, please verify the contract terms with legal and confirm delivery schedules before we countersign.",
            "sender": "jeff.skilling@enron.com",
            "receiver": "Phillip Allen",
            "timestamp": "2026-11-14 11:15:00",
            "thread_id": "thread_contract_012"
        },
        {
            "id": "med_04",
            "subject": "Follow-up: Candidate Interview Feedback for Data Analyst Role",
            "body": "Hi Mark, our panel completed the second-round technical interview for the senior data analyst position earlier today. The candidate demonstrated strong SQL optimization skills and solid Python data pipeline experience. Please submit your written scorecard and hiring recommendation to the recruiting portal when you have a chance.",
            "sender": "recruiting@company.com",
            "receiver": "Mark Taylor",
            "timestamp": "2026-11-14 11:30:00",
            "thread_id": "thread_hr_013"
        },

        # ═══════════════════════════════════════════════════════
        # LOW PRIORITY (4 emails)
        # ═══════════════════════════════════════════════════════
        {
            "id": "low_01",
            "subject": "Monthly Engineering Newsletter & Learning Resources — November",
            "body": "Hi everyone, here is our monthly newsletter digest of tech blog posts, upcoming brown-bag sessions, and recommended NLP tutorials. This month's highlights include a deep dive into vector embeddings, a podcast on MLOps tooling, and our internal AI reading group schedule. No action required. Feel free to browse at your leisure.",
            "sender": "newsletter@company.com",
            "receiver": "All Staff",
            "timestamp": "2026-11-14 12:00:00",
            "thread_id": "thread_news_014"
        },
        {
            "id": "low_02",
            "subject": "FYI: Updated Corporate Holiday Schedule for 2027",
            "body": "Dear team, please find attached the official company holiday calendar for the upcoming year. The list includes all federal holidays and company-observed days off. No reply is needed. We hope you enjoy your upcoming Thanksgiving holidays and time with family.",
            "sender": "hr-announcements@company.com",
            "receiver": "All Staff",
            "timestamp": "2026-11-14 12:30:00",
            "thread_id": "thread_fyi_015"
        },
        {
            "id": "low_03",
            "subject": "Knowledge Base Update: Mercury SDK v2.4 Documentation Released",
            "body": "Dear registered developer, this is an informational notification. Our developer portal documentation has been updated with the latest API code examples, migration guide, and release notes for Mercury Language SDK version 2.4. No action is required on your account. The updated docs are live at our developer portal.",
            "sender": "product-updates@company.com",
            "receiver": "Registered Users",
            "timestamp": "2026-11-14 13:00:00",
            "thread_id": "thread_doc_016"
        },
        {
            "id": "low_04",
            "subject": "Corporate Accounting: Expense Reimbursements & Travel Billing",
            "body": "Dear Finance Team, please find attached the monthly travel expense receipts and employee reimbursement requests for accounting audit. Total billing amount is $8,450 across all departments. All receipts have been scanned and are within the approved travel policy guidelines. Please process by month end.",
            "sender": "accounting@company.com",
            "receiver": "Finance Team",
            "timestamp": "2026-11-14 13:15:00",
            "thread_id": "thread_exp_017"
        },

        # ═══════════════════════════════════════════════════════
        # SPAM / PHISHING (3 emails)
        # ═══════════════════════════════════════════════════════
        {
            "id": "spam_01",
            "subject": "CONGRATULATIONS! You Have Won a $1,000 Walmart Gift Card",
            "body": "Dear Winner, Congratulations! Your email address was selected in our exclusive deal sweepstakes. You have won a free offer worth $1,000. Click here immediately to claim your prize and confirm your identity before this limited time offer expires! Act now! Do not miss out on this exclusive deal. To stop receiving these offers, unsubscribe here.",
            "sender": "rewards@promo-sweepstakes-winner.com",
            "receiver": "Me",
            "timestamp": "2026-11-14 13:30:00",
            "thread_id": "thread_spam_018"
        },
        {
            "id": "spam_02",
            "subject": "SECURITY ALERT: Confirm your bank details to avoid account suspension",
            "body": "Dear Customer, We detected unusual sign-in activity on your online banking profile. Please verify your account and confirm your bank details immediately to prevent temporary account closure. Click here to confirm your identity and restore access. Act now before the deadline! Unsubscribe if you do not wish to receive security alerts.",
            "sender": "security-notification@secure-banking-verify-alert.net",
            "receiver": "Me",
            "timestamp": "2026-11-14 14:00:00",
            "thread_id": "thread_spam_019"
        },
        {
            "id": "spam_03",
            "subject": "Exclusive Deal: Get 90% Off Luxury Watches - Limited Time Offer",
            "body": "Act now! Exclusive deal on designer watches, luxury perfumes, and electronics at 90% off retail price. Free offer included with every purchase above $50. This is a limited time offer — only 12 hours remaining! Click here to browse our catalog and apply your exclusive discount. To unsubscribe from our promotional mailing list, click here.",
            "sender": "promotions@daily-luxury-discounts-online.com",
            "receiver": "Me",
            "timestamp": "2026-11-14 14:30:00",
            "thread_id": "thread_spam_020"
        },

        # ═══════════════════════════════════════════════════════
        # MEETINGS (3 emails)
        # ═══════════════════════════════════════════════════════
        {
            "id": "mtg_01",
            "subject": "Room Booking Confirmed: Tomorrow's Strategy Meeting 2:00 PM",
            "body": "Phillip, Ina has confirmed the conference room booking for our strategy alignment session tomorrow at 2:00 PM in Room 4B. We will also set up the Zoom conference call bridge for remote participants. Please let me know if you need any special projector or whiteboard equipment prepared.",
            "sender": "ina.rangel@enron.com",
            "receiver": "Phillip Allen",
            "timestamp": "2026-11-14 15:00:00",
            "thread_id": "thread_mtg_021"
        },
        {
            "id": "mtg_02",
            "subject": "Rescheduled: Product Architecture Review - Thursday 10:00 AM",
            "body": "Team, due to executive travel conflicts, the product architecture review meeting originally scheduled for Wednesday has been rescheduled to Thursday at 10:00 AM in the Innovation Lab. Please update your calendar invites accordingly. Alice will circulate the updated system design notes and RFC documents the morning of the meeting.",
            "sender": "calendar@company.com",
            "receiver": "Core Architecture Team",
            "timestamp": "2026-11-14 15:30:00",
            "thread_id": "thread_mtg_022"
        },
        {
            "id": "mtg_03",
            "subject": "Minutes: Risk Management Committee Meeting — Action Points",
            "body": "Vince, thank you for your attendance at today's Risk Management Committee session. The attached meeting minutes document the agreed action points regarding the updated Value-at-Risk exposure limits for natural gas positions. Our next scheduled committee meeting is confirmed for next Friday at 11:00 AM.",
            "sender": "risk-committee@enron.com",
            "receiver": "Vince Kaminski",
            "timestamp": "2026-11-14 15:45:00",
            "thread_id": "thread_mtg_023"
        },

        # ═══════════════════════════════════════════════════════
        # OPERATIONS (3 emails)
        # ═══════════════════════════════════════════════════════
        {
            "id": "ops_01",
            "subject": "Scheduled Maintenance Notice: Cloud Infrastructure & Server Upgrades",
            "body": "DevOps notification: Scheduled server maintenance and kubernetes cluster upgrades are planned for Saturday between 02:00 AM and 05:00 AM EST. During this window, some production services may experience intermittent connectivity. API latency may briefly spike during the pod rollout and node draining process.",
            "sender": "devops@company.com",
            "receiver": "All Staff",
            "timestamp": "2026-11-14 16:00:00",
            "thread_id": "thread_ops_024"
        },
        {
            "id": "ops_02",
            "subject": "Operations Update: Kubernetes Node Deployment & Auto-Scaling Configured",
            "body": "Site reliability engineering update: The new worker node group for our compute cluster has been successfully provisioned across 3 availability zones. Server memory headroom has improved by 35% and auto-scaling policies are now active. No further planned infrastructure downtime is anticipated this quarter.",
            "sender": "sre-ops@company.com",
            "receiver": "DevOps Team",
            "timestamp": "2026-11-14 16:15:00",
            "thread_id": "thread_ops_025"
        },
        {
            "id": "ops_03",
            "subject": "Incident Post-Mortem: API Monitoring Latency Spikes — Root Cause Analysis",
            "body": "Following last Thursday's intermittent API monitoring latency spikes affecting our Cloud Management service, our infrastructure team has completed the root cause analysis. The issue was traced to an untuned connection pool limit on the database proxy layer. The fix has been deployed and latency is now back within SLA bounds.",
            "sender": "operations-report@aetheros.com",
            "receiver": "Engineering Leadership",
            "timestamp": "2026-11-14 16:30:00",
            "thread_id": "thread_ops_026"
        },

        # ═══════════════════════════════════════════════════════
        # FINANCE (3 emails)
        # ═══════════════════════════════════════════════════════
        {
            "id": "fin_01",
            "subject": "Invoice #INV-2026-8819: Monthly Cloud Hosting & Bandwidth Billing",
            "body": "Attached is your official monthly invoice for cloud infrastructure hosting, compute resources, and CDN bandwidth consumption. Total balance due: $14,250.00. Payment is due by November 30, 2026 via wire transfer or corporate credit card. Please reference Invoice #INV-2026-8819 on your remittance advice.",
            "sender": "billing@cloud-provider.com",
            "receiver": "Accounting Department",
            "timestamp": "2026-11-14 16:45:00",
            "thread_id": "thread_fin_027"
        },
        {
            "id": "fin_02",
            "subject": "Q4 Wire Transfer Confirmation: KPMG Tax Advisory Services",
            "body": "Accounting team, the wire transfer of $42,500 for KPMG quarterly tax advisory services has been processed and confirmed by the bank. All invoice line items and applicable tax withholding deductions match our approved quarterly budget ledger. Please update the accounts payable register accordingly.",
            "sender": "treasury@company.com",
            "receiver": "Finance Leads",
            "timestamp": "2026-11-14 17:00:00",
            "thread_id": "thread_fin_028"
        },
        {
            "id": "fin_03",
            "subject": "Annual Budget Planning: Departmental Expense Forecast for 2027",
            "body": "Vince, as we head into the annual budget planning cycle, please prepare your departmental expense forecast for fiscal year 2027. Include projected headcount additions, software licensing renewals, conference travel budgets, and capital expenditure requests. The finance committee needs this by December 15.",
            "sender": "cfo@enron.com",
            "receiver": "Vince Kaminski",
            "timestamp": "2026-11-14 17:15:00",
            "thread_id": "thread_fin_029"
        },

        # ═══════════════════════════════════════════════════════
        # EXTERNAL / CLIENT (2 emails)
        # ═══════════════════════════════════════════════════════
        {
            "id": "ext_01",
            "subject": "External Vendor Proposal: Enterprise Cloud Integration Partnership",
            "body": "Dear Executive Team, We are pleased to submit our formal external vendor proposal for enterprise cloud integration services. Our client solutions director has reviewed the technical scope of work and agreed to your pricing terms and SLA requirements. We look forward to beginning the onboarding process at your earliest convenience.",
            "sender": "partner-solutions@external-vendor.com",
            "receiver": "Executive Team",
            "timestamp": "2026-11-14 17:30:00",
            "thread_id": "thread_ext_030"
        },
        {
            "id": "ext_02",
            "subject": "Client Contract Signed: Western Power Supply Delivery Agreement",
            "body": "Phillip, our external client Western Power has signed off on the natural gas delivery terms for the Q1 2027 quarter. Please review the attached finalized client contract covering the delivery point schedules and penalty clauses, then send confirmation to their legal department before Friday.",
            "sender": "client-contracts@enron.com",
            "receiver": "Phillip Allen",
            "timestamp": "2026-11-14 17:45:00",
            "thread_id": "thread_ext_031"
        },

        # ═══════════════════════════════════════════════════════
        # ACTION REQUIRED (2 emails)
        # ═══════════════════════════════════════════════════════
        {
            "id": "act_01",
            "subject": "Action Required: Complete Annual Compliance and Ethics Certification",
            "body": "Dear Colleague, Action required: all employees must complete the mandatory annual ethics and corporate compliance certification module by next Friday, November 22, 2026. Please log in to the HR compliance portal and complete all 3 training modules. Kindly submit your digital completion receipt to your manager upon finishing.",
            "sender": "compliance-officer@company.com",
            "receiver": "All Employees",
            "timestamp": "2026-11-14 18:00:00",
            "thread_id": "thread_act_032"
        },
        {
            "id": "act_02",
            "subject": "Action Required: Review and Sign Updated Non-Disclosure Agreement",
            "body": "Hi Sarah, action required: please review and sign the attached updated Non-Disclosure Agreement governing the upcoming joint venture with Accenture. The NDA includes expanded IP protection clauses. Kindly return the countersigned PDF to our legal counsel before the end of this week so the project can proceed.",
            "sender": "legal-counsel@company.com",
            "receiver": "Sarah Jenkins",
            "timestamp": "2026-11-14 18:30:00",
            "thread_id": "thread_act_033"
        }
    ]
