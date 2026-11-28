"""Small fictional policy handbook corpus for local demonstrations."""

from .policy_models import PolicyDocument, PolicySection

SYNTHETIC_POLICY_CORPUS = (
    PolicyDocument(
        source_id="SYN-POLICY-SCHED-001",
        title="Fictional North Clinic Scheduling Guide",
        sections=(
            PolicySection(
                name="Appointment Changes",
                text=(
                    "At the fictional North Clinic, appointment changes should be "
                    "requested at least one business day before the scheduled visit. "
                    "Requests received later are routed to a scheduling coordinator "
                    "for review. This sample rule applies only to this fictional guide."
                ),
            ),
            PolicySection(
                name="Available Appointment Information",
                text=(
                    "Staff may share available appointment times, specialty, and "
                    "fictional clinic location. An availability lookup does not reserve "
                    "or book a visit; a coordinator must confirm any scheduling action."
                ),
            ),
        ),
    ),
    PolicyDocument(
        source_id="SYN-POLICY-REF-001",
        title="Fictional Referral Coordination Guide",
        sections=(
            PolicySection(
                name="Referral Intake",
                text=(
                    "A fictional referral record includes its synthetic referral "
                    "identifier, requested specialty, referring department, and the "
                    "receiving clinic. If required administrative details are missing, "
                    "route the record to a human coordinator instead of guessing."
                ),
            ),
            PolicySection(
                name="Referral Status",
                text=(
                    "Referral status is read from the matching synthetic record. "
                    "An unavailable identifier is reported as not found and is not "
                    "treated as evidence that a referral was completed."
                ),
            ),
        ),
    ),
    PolicyDocument(
        source_id="SYN-POLICY-INS-001",
        title="Fictional Insurance Plan Reference",
        sections=(
            PolicySection(
                name="Synthetic Specialist Copays",
                text=(
                    "For this fictional reference only, SYN-PLAN-12 lists a "
                    "35 dollar specialist visit copay and SYN-PLAN-24 lists a "
                    "30 dollar specialist visit copay. These examples are not real "
                    "benefit terms and do not confirm a member's eligibility."
                ),
            ),
            PolicySection(
                name="Eligibility Lookup",
                text=(
                    "Eligibility information must be retrieved from the matching "
                    "synthetic member fixture. Handbook text alone does not establish "
                    "that a member is eligible or that a service is covered. Refer "
                    "unavailable records to a human coordinator."
                ),
            ),
        ),
    ),
)
