from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

cdr_knowledge = """
INTERNATIONAL CALL IDENTIFICATION:
- A call is international if the CdPN (Called Party Number) starts with a country code other than the local country.
- Check the CLI (Calling Line Identity) prefix to determine origin country.
- If CDE = 999, this typically indicates an international routing flag.
- VPN group and site name can help confirm if the call traversed international trunks.

SPEARLINE TEST PROCEDURE:
- When an international call fails or has quality issues, first step is to test the called number via Spearline.
- Spearline isolates whether the issue is on the called number side or the network/trunk side.
- If Spearline confirms the number is reachable and clear, the issue is on your network side.
- If Spearline fails, raise a ticket with the carrier for the destination number.

FROM NUMBER ANALYSIS:
- Check CLI field in the CDR — this is the originating number.
- Compare CLI against the default site number for that trunk group (T1T7 field).
- If CLI does not match the default site number, it may be a diversion or PAI manipulation.
- PAI field in CDR shows the asserted identity — if it differs from CLI, a diversion is likely.
- Check Privacy field — if set to something other than 'none', CLI may be withheld intentionally.

TRACE PULLING PROCEDURE:
- Pull a SIP trace from the SBC (AudioCodes) for the CID (Call ID) shown in the CDR.
- Look for the SIP INVITE, 180 Ringing, 200 OK, and BYE messages.
- Check RC (Release Cause) field — Normal Call Clearing (016) means the call ended cleanly by one party.
- Check LegendRCHistory for the SIP BYE reason and location code.
- location=TPO means the call was released from the outbound/terminating side.
- If you see 4xx or 5xx SIP codes in the trace, escalate based on the error code.

REROUTE PROCEDURE:
- Only perform a reroute after: Spearline test done, FROM number verified, SIP trace reviewed.
- Reroute means changing the outbound trunk or routing rule to use an alternate carrier path.
- Document the CID, original route, and new route before making changes.
- After reroute, monitor the next 3-5 calls to the same destination to confirm improvement.

CDR FIELD REFERENCE:
- CLI: Calling Line Identity (who is calling)
- CdPN: Called Party Number (who is being called)
- PAI: P-Asserted-Identity (network-asserted identity, may differ from CLI)
- CDE: Call Detail Extension code
- RC: Release Cause
- T1T7: Trunk group identifier
- CID: Call ID (unique identifier for SIP trace lookup)
- deta/dson: Post-dial delay metrics
- Duration: Call duration in seconds
- VPN: Virtual Private Network group

GUIDE TO ERRORS:
- Invalid calling number : The number is not configured , the site is closed , the Trunk doesn't have a default site 
- Invalid called number : the number is canceled , the number is free , BTD issue .
- UPAC is not identified :  the pre analysis list on the site is not matching the called number go check it out from the hookah 
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
