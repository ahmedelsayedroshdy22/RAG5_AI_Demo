from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

cdr_knowledge = """

You are a BTG voice network assistant. Answer ONLY using the provided context below.
If the answer is not found in the context, respond with:
"I don't have specific information about that in my knowledge base. Please check with the your senior"
Do NOT invent definitions or explanations not present in the context.




WHAT IS BUSINESS TALK GLOBE (BTG):
- BTG uses Orange's global presence to carry international and local voice calls.
- Covers 154 countries and territories globally, with full local offer in 24 countries.
- Single voice provider for both local and international services.
- Average 18-20% overall savings on voice spend for customers.
- Full-IP network, SIP-enabled, ready for UC (Unified Communications).

BTG SERVICES:
- VOMO: Local Voice Service.
- BTIP: Business Talk Over Internet, site-to-site, domestic and international calls.
- CCA: Contact Center Access Numbers.
- BTD: Business Talk Digital, self-service provisioning portal for customers.
- On-net: Calls between customer sites through Orange network.
- Off-net: Calls terminating outside Orange network, used for international breakout.

TYPE OF CONNECTIONS:
- TDM: Direct TDM connection via dedicated leased line.
- IP: SIP-based IP trunking to connect IP PBX to Business Talk.
- Certified IP PBX vendors: Cisco, Avaya, Alcatel, Microsoft, Mitel, Siemens.

KEY INFRASTRUCTURE COMPONENTS:
- IP PBX: Core telephony element at the customer site.
- CE: Customer Edge, router provided by Orange to connect the customer to the network.
- PE: Provider Edge, Orange backbone router connected to the CE.
- SBC: Session Border Controller, security element protecting customer and Orange networks.
- Hookah: The Brain, Orange internal routing and CDR platform.

HOOKAH - THE BRAIN:
- Hookah is the central intelligence of the BTG voice network.
- Hardware and software historically provided by HP and Atos Origin (SEP).
- Since 2017 hardware and software provided by Atos only. Newer deployments use VHookah.
- Hookah sits at Stage 3 of the call flow between the POP and the terminating site.

HOOKAH FUNCTIONS:
- Address Resolution: Resolves destination numbers to correct routing paths.
- Routing Decisions: Decides how and where to route each call.
- SIP Header Modifications: Modifies CLI, CdPN, PAI and other SIP headers.
- CDR Generation: Produces Call Detail Records for every call passing through the network.

CALL FLOW STAGES:
- Stage 1 - Collecting Carrier: First interface for the OBS network. Collects inbound traffic and sends to first POP. Device types are NEO (OPM soft switch) or CL4 SBC.
- Stage 2 - POP (Point of Presence): Orange network entry point. Passes upstream traffic to Hookah.
- Stage 3 - Hookah: Receives upstream from POP, applies routing and SIP modifications, delivers downstream to the terminating site.
- Stage 4 - Terminating Site: Final destination which can be on-net or off-net customer site or call center.

CALL FLOW TYPES:
- On-net to On-net: Customer calls between own sites through Orange network.
- On-net to Off-net: Customer uses Orange network to make international or PSTN calls.
- BTLVS: Call collected by carrier, passed through BTG, terminated at customer site.
- Call Collection: Inbound call collected from PSTN or carrier, through BTG, through terminating carrier, to destination agent.
- Offnet Termination: Customer site, through OBS SBC, through BTG, through Offnet carrier, to destination.

SBC TYPES IN ORANGE:
- aSBC (Access SBC): Dedicated to customers requiring secured SIP Trunking. 12 aSBCs plus BTOI aSBCs in network.
- eSBC (NBI Carrier SBC): Dedicated to carriers requiring SIP Trunking. 8 eSBCs in network.

aSBC REGIONAL PROXIES:
- EUR: Main proxy JFRA531 on PE BFRA518, Backup proxy JQQT531 on PE BLCY519.
- AMR: Main proxy JLAX531 on PE BLAX810, Backup proxy JJRE531 on PE BJRE519.
- APA: Main proxy JXKT531 on PE BXKT619, Backup proxy JXSP531 on PE BXSP618.

eSBC LOCATIONS:
- AME: JIAD620 and JLAX620.
- EMA: JZRB620 and JLBG620.
- APA: JMEL620, JOKO620, JOSA620, JXSP620.

SBC VENDORS:
- ACME SBCs are used for aSBCs (Access SBCs).
- Ribbon SBCs are used for eSBCs (NBI Carrier SBCs).
- Other vendors that exist: FRAFOS, INGATE System.

T1T7 IDENTITY:
- T1T7 is the trunk group identifier in the CDR showing which SBC handled the call.
- Primary SBC values: 04 and 08.
- Secondary SBC values: 14 and 18.
- Load Balancing values: 24 and 28.
- Compare the CLI in the CDR against the default site number for that T1T7 to detect diversions.

CCA - CONTACT CENTER ACCESS SERVICE:
- Traffic is collected from different countries via Access Numbers and terminated to the customer call center.
- PSTN: Can be dialed from anywhere.
- DTFN: Domestic Toll Free Number, domestic calls only.
- ITFS: International Toll Free Service.
- UIFN: Universal International Free-phone Number.

BTD - BUSINESS TALK DIGITAL:
- Self-service customer portal for real-time online ordering and provisioning.
- Used for change management, profile management, and business continuity actions.
- When a CDR shows a BTD issue under Invalid Called Number, the number may be canceled, unallocated, or has a provisioning problem.
- Action: Check the number status in the BTD portal and verify provisioning.

CDR TOOL:
- CDR Web Portal URL: https://cdrrepo.rp-ocn.apps.ocn.infra.ftgroup/cdrwebapp/
- CDRs are generated by Hookah for every call passing through the BTG network.

INTERNATIONAL CALL IDENTIFICATION:
- A call is international if the CdPN (Called Party Number) starts with a country code other than the local country.
- Check the CLI (Calling Line Identity) prefix to determine origin country.
- If CDE = 999, this typically indicates an international routing flag.
- VPN group and site name can help confirm if the call traversed international trunks.

CDR FIELD REFERENCE:
- CLI: Calling Line Identity (who is calling, the originating number).
- CdPN: Called Party Number (who is being called, the destination number).
- PAI: P-Asserted-Identity (network-asserted identity, may differ from CLI if diversion occurred).
- CDE: Call Detail Extension code. CDE=999 typically indicates an international routing flag.
- RC: Release Cause. RC=016 means Normal Call Clearing, call ended cleanly.
- T1T7: Trunk group and SBC identifier.
- CID: Call ID, unique identifier used to pull SIP trace from the SBC.
- deta/dson: Post-dial delay metrics. Abnormal values indicate routing or PDD issues.
- Duration: Call duration in seconds. Duration=0 usually means the call never connected.
- VPN: Virtual Private Network group, confirms if call traversed international trunks.
- Privacy: If not set to none, CLI may be intentionally withheld by the originating party.

SPEARLINE TEST PROCEDURE:
- When an international call fails or has quality issues, first step is to test the called number via Spearline.
- Spearline isolates whether the issue is on the called number side or the network and trunk side.
- If Spearline confirms the number is reachable and clear, the issue is on your network side. Pull SIP trace and check the trunk.
- If Spearline fails, raise a ticket with the carrier for the destination number.
- If Spearline is unavailable, attempt a manual test call from an alternate carrier path and document the result before escalating.

FROM NUMBER ANALYSIS:
- Check CLI field in the CDR, this is the originating number.
- Compare CLI against the default site number for that trunk group (T1T7 field).
- If CLI does not match the default site number, it may be a diversion or PAI manipulation.
- PAI field in CDR shows the asserted identity. If it differs from CLI, a diversion is likely such as a call forward.
- Check Privacy field. If set to something other than none, CLI may be withheld intentionally.
- Diversion means the call was forwarded from another number before reaching the trunk.
- Action: Trace the forwarding chain and verify if PAI manipulation is expected or anomalous.

TRACE PULLING PROCEDURE:
- Pull a SIP trace from the SBC (AudioCodes, ACME, or Ribbon) for the CID (Call ID) shown in the CDR.
- Look for the SIP INVITE, 180 Ringing, 200 OK, and BYE messages.
- Check RC (Release Cause) field. Normal Call Clearing (016) means the call ended cleanly by one party.
- Check LegendRCHistory for the SIP BYE reason and location code.
- location=TPO means the call was released from the outbound or terminating side.
- location=TPI means the call was released from the inbound or originating side.
- 4xx SIP codes are client-side errors such as 404 Not Found, 486 Busy, 408 Timeout. Check called number and config.
- 5xx SIP codes are server-side errors such as 503 Service Unavailable. Escalate to carrier or network team.
- 6xx SIP codes are global failures such as 603 Decline. The called party rejected the call.

REROUTE PROCEDURE:
- Only perform a reroute after Spearline test is done, FROM number is verified, and SIP trace is reviewed.
- Reroute means changing the outbound trunk or routing rule to use an alternate carrier path.
- Document the CID, original route, and new route before making changes.
- Apply the routing change.
- Monitor the next 3-5 calls to the same destination via CDR tool and live trace.
- Confirm improvement before closing the case.

UPAC ERROR:
- UPAC stands for a pre-analysis routing identifier used in Hookah.
- "UPAC is not identified" error means the pre-analysis list on the site 
  is not matching the called number.
- Action: Go to Hookah and update the pre-analysis list for that site.

GUIDE TO ERRORS:
- Invalid calling number: The number is not configured, the site is closed, the trunk does not have a default site.
- Invalid called number: The number is canceled, the number is unallocated or free, BTD issue.
- UPAC is not identified: The pre-analysis list on the site is not matching the called number. Go check and update it from Hookah.
"""

with open("cdr_knowledge.txt", "w") as f:
    f.write(cdr_knowledge)

loader = TextLoader("cdr_knowledge.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="cdr_vectorstore"
)

print(f"✅ Vector store created with {len(chunks)} chunks.")
