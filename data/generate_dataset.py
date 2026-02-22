"""
generate_dataset.py
--------------------
Generates a synthetic labeled dataset of 200 text samples for the
AI-Based Digital Content Manipulation Risk Detector.

Labels:
  - Low      : Neutral, factual content
  - Moderate : Emotional, persuasive content
  - High     : Fear-based, urgency-based, sensational content
"""

import csv
import os
import random

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

# ── Sample pools (≈67 per class → 201 → we'll trim to 200) ───────────────────

LOW_RISK_SAMPLES = [
    "The stock market closed slightly higher today amid moderate trading volume.",
    "Scientists have discovered a new species of frog in the Amazon rainforest.",
    "The city council met yesterday to discuss the upcoming budget for road repairs.",
    "A local library launched a new digital reading program for children.",
    "Weather forecasters predict mild temperatures across the region this week.",
    "The governor signed a bill to increase funding for public schools.",
    "Researchers published findings on the migratory patterns of monarch butterflies.",
    "A tech company announced quarterly earnings that met analyst expectations.",
    "The museum opened a new exhibit on ancient Egyptian artifacts.",
    "Public transportation ridership increased by 5% compared to last year.",
    "Scientists completed a 10-year study on deep-sea coral reef ecosystems.",
    "The university announced scholarships for students in STEM programs.",
    "Local farmers reported an average harvest season due to typical rainfall.",
    "The central bank held interest rates steady at its quarterly meeting.",
    "A new community garden was planted on the city's east side.",
    "Astronomers observed a distant galaxy using the latest space telescope.",
    "The hospital completed renovations to its emergency wing last Tuesday.",
    "A report shows that air quality in the city improved over the past decade.",
    "The national park service released updated trail maps for hikers.",
    "Inflation figures came in within the expected range for this quarter.",
    "Road maintenance crews will work on Highway 12 starting next Monday.",
    "An academic conference on renewable energy was held in the capital.",
    "The census bureau released updated population estimates for the state.",
    "A local nonprofit organized a weekend food drive for community members.",
    "The legislature passed a routine bill extending infrastructure grants.",
    "Marine biologists documented new behavioral patterns in humpback whales.",
    "A software company released a minor update to its project management tool.",
    "The transportation department started a pilot program for electric buses.",
    "Global temperatures last month were within the historically normal range.",
    "A book on climate resilience strategies won this year's science award.",
    "The city announced plans to repave several residential streets next month.",
    "Archaeologists uncovered pottery shards dating back 2,000 years near the river.",
    "International trade talks are scheduled to resume in Geneva next week.",
    "A local school won a regional robotics competition last Saturday.",
    "The health department reminded residents about upcoming flu vaccine clinics.",
    "Officials confirmed that water quality tests passed all safety standards.",
    "A new pedestrian bridge connecting two neighborhoods was inaugurated.",
    "The county library extended its operating hours to accommodate more visitors.",
    "An environmental study found stable bird population levels in wetland areas.",
    "The telecommunications company upgraded its fiber network in rural districts.",
    "A community college offered free coding bootcamps for unemployed adults.",
    "The annual wildfire safety report indicated preparedness is on schedule.",
    "A shipping port reported record cargo tonnage for standard commodities.",
    "Regulators approved a routine license renewal for a regional power plant.",
    "The school board voted to update its curriculum guidelines for next year.",
    "A local theater hosted a youth arts festival last weekend.",
    "An annual report on childhood literacy showed slight improvement nationally.",
    "Construction on the new sports complex is expected to finish in spring.",
    "Officials released a map of proposed bike-lane extensions across the city.",
    "Research on soil quality improvements in farmland was published this month.",
    "The regional transit authority approved a modest fare adjustment.",
    "A tech startup was accepted into a federally funded innovation program.",
    "The weather service confirmed that last winter was about average in snowfall.",
    "A new trail in the state park opened after months of construction work.",
    "The financial regulator issued routine guidelines for small business lending.",
    "A documentary film about deep-sea exploration premiered at a science festival.",
    "Local elections are scheduled for the third Tuesday next month.",
    "The agriculture department released guidelines for responsible pesticide use.",
    "A nonprofit launched a mentorship program connecting students with engineers.",
    "Engineers completed a routine inspection of the city's main water dam.",
    "A public hearing on the proposed zoning changes was held last Thursday.",
    "The housing authority released its annual affordable housing progress report.",
    "Volunteer crews planted 500 trees along the city's riverbank last weekend.",
    "A partnership between two universities will support joint climate research.",
    "The tourism board published updated visitor statistics for the past quarter.",
    "A committee reviewed and approved standard safety codes for new buildings.",
    "Officials confirmed that fire hydrant inspections are proceeding on schedule.",
]

MODERATE_RISK_SAMPLES = [
    "You won't believe what this politician said that is making everyone furious!",
    "This heartbreaking story will make you question everything you believed.",
    "Scientists are finally admitting what many people suspected for years.",
    "The government doesn't want you to know these shocking economic truths.",
    "Parents are outraged after a school made a decision affecting all students.",
    "The real reason prices keep rising that nobody is talking about.",
    "This viral video shows the uncomfortable truth about modern media.",
    "Why millions of people are abandoning traditional news sources right now.",
    "Experts are divided on this controversial treatment approach — who is right?",
    "The disturbing trend in social media usage that affects our children.",
    "People are waking up to the manipulation happening right in front of them.",
    "This emotional story of a family torn apart by politics will move you.",
    "Are our children's textbooks hiding the truth from them? Many parents say yes.",
    "The pharmaceutical company they said was safe is now facing major lawsuits.",
    "One small town's battle against corporate greed is inspiring thousands.",
    "Why everyday people no longer trust the mainstream narrative on health.",
    "The shocking survey results showing how democracy is being silently eroded.",
    "Medical professionals are raising alarms, but nobody seems to be listening.",
    "This banned documentary reveals the dark side of the food industry.",
    "Citizens are demanding answers after the government chose silence over truth.",
    "A brave whistleblower faces career destruction for speaking the truth in public.",
    "The alarming increase in stress and anxiety is a symptom of a broken system.",
    "Why more families are choosing to homeschool than ever before in history.",
    "The hidden agenda behind the new education reforms exposed by insiders.",
    "Millions are losing faith in institutions that were once considered pillars of society.",
    "This powerful letter from a nurse is going viral and you need to read it.",
    "The emotional toll of financial insecurity is tearing communities apart silently.",
    "Why we should be deeply worried about the concentration of media power.",
    "A mother's tearful plea for policy change is being ignored by lawmakers.",
    "The subtle manipulation in advertising that preys on your deepest insecurities.",
    "People feeling ignored deserve to know the truth behind these policy decisions.",
    "This investigative report challenges everything we thought we knew about safety.",
    "The uncomfortable statistics on inequality that officials prefer not to discuss.",
    "Local heroes are fighting a system that seems stacked against working families.",
    "Why so many young voters feel betrayed by both major political parties today.",
    "The hidden psychological effects of social media on teenagers exposed at last.",
    "A growing number of citizens believe change is overdue and are demanding it.",
    "The surprising way government agencies shape the news stories you consume.",
    "This deeply moving account of injustice shows why reform is desperately needed.",
    "Researchers are questioning the official stance on a widely prescribed drug.",
    "Why people are angry: the systemic failures that leaders keep sweeping aside.",
    "The secret corporate lobbying that has stalled meaningful climate legislation.",
    "This chilling exposé on data privacy reveals just how vulnerable we all are.",
    "An open letter from teachers is going viral, and it should make us all think.",
    "The powerful economic forces pushing ordinary families further into poverty.",
    "Citizens are right to question the motivations of those in power right now.",
    "The real cost of inaction on mental health that society refuses to acknowledge.",
    "Why the housing crisis is forcing a generation to rethink the American dream.",
    "A senator's leaked communication reveals a troubling disregard for constituents.",
    "The media's obsession with distraction is helping powerful interests stay hidden.",
    "This eye-opening graph about income inequality will leave you speechless today.",
    "What they told us about nutrition for decades is now being challenged by science.",
    "People are fed up with being talked down to by elites who live differently.",
    "The manipulation of public opinion has never been as sophisticated as it is now.",
    "Why increasing numbers of citizens are choosing to protest rather than vote.",
    "The quiet erosion of civil liberties should concern every thoughtful citizen.",
    "This underreported story about environmental damage deserves national attention.",
    "A town fighting back against pharmaceutical opioid distribution is winning.",
    "Why the gap between the rich and everyone else keeps widening every year.",
    "The troubling evidence suggests important safety standards are being ignored.",
    "Parents are questioning why schools adopted this curriculum without community input.",
    "The way social networks amplify outrage is reshaping political participation.",
    "Why millions believe the system is fundamentally broken and needs overhaul.",
    "The financial industry's influence over government hasn't ever been stronger.",
    "This firsthand account of poverty challenges comfortable middle-class assumptions.",
    "Doctors are speaking out against pressures they face from hospital administrators.",
]

HIGH_RISK_SAMPLES = [
    "URGENT: The government is secretly poisoning the water supply — act NOW!",
    "BREAKING: Scientists confirm the end of civilization is imminent — prepare yourself!",
    "WARNING: 5G towers being used to inject microchips into the population secretly!",
    "ALERT: Dangerous chemicals found in ALL major food brands — stop eating them immediately!",
    "SHOCKING REVELATION: The elite are planning mass population reduction this year!",
    "DO NOT GET VACCINATED: Insider confirms vaccines contain tracking nano-devices!",
    "EMERGENCY WARNING: Banks will collapse in 72 hours — withdraw your money NOW!",
    "EXPOSED: The cure for cancer has been suppressed by Big Pharma for decades!",
    "CRITICAL ALERT: Foreign agents have infiltrated every level of government!",
    "PANIC NOW: Scientists hid data showing the poles will shift catastrophically!",
    "TRUTH BOMB: The moon landing was faked and this video proves it beyond doubt!",
    "SHARE BEFORE IT'S DELETED: The mainstream media's greatest lie exposed forever!",
    "THEY DON'T WANT YOU TO KNOW: Chemtrails are causing mass sterilization globally!",
    "THE GREAT RESET IS REAL: Elites planning to take away all private property!",
    "WARNING: Eating microwave food causes instant DNA damage — stop immediately!",
    "FINAL COUNTDOWN: The New World Order will seize control by end of the month!",
    "EXPOSED: Hospitals are killing patients to inflate pandemic death toll numbers!",
    "LAST CHANCE: Government preparing to shut down internet access indefinitely!",
    "URGENT TRUTH: This common household chemical is being used for mass mind control!",
    "CRISIS ALERT: Deep state operatives are preparing a massive false flag event!",
    "BREAKING: Scientists silenced after discovering vaccines alter human DNA forever!",
    "THEY ARE LYING: COVID was engineered in a lab to depopulate the earth!",
    "SHOCKING: Drinking tap water causes immediate neurological damage, experts confirm!",
    "ALERT: The government is tracking every citizen through smart TV microphones!",
    "FORBIDDEN KNOWLEDGE: Only the elite know this ancient cure for all diseases!",
    "ACT NOW: Global elites meeting this week to finalize full enslavement of mankind!",
    "URGENT: Your phone is broadcasting your location to foreign intelligence agencies!",
    "WARNING: This food additive is causing a secret epidemic of mental illness!",
    "EXPOSED: Military drones are conducting surveillance on every American home!",
    "BREAKING: Scientists confirm apocalyptic meteor will hit Earth within 30 days!",
    "THE TRUTH SUPPRESSED: Antarctica contains the gateway to a secret underground civilization!",
    "CRITICAL: Fluoride in water is a mind-control chemical proven by leaked docs!",
    "FINAL WARNING: The deep state plans to detonate an EMP to end modern civilization!",
    "SHARE WIDELY: This banned video reveals how billionaires control all elections!",
    "SHOCKING PROOF: World leaders are controlled by an alien shadow government!",
    "EMERGENCY: Central banks plan to eliminate cash and enslave you digitally by 2025!",
    "DO NOT COMPLY: New health mandates are designed to permanently impoverish citizens!",
    "EXPOSED: Big Tech is erasing your memory by manipulating your daily news feed!",
    "THEY WANT YOU SICK: Pharmaceutical companies are actively blocking cheap cures!",
    "URGENT REVELATION: Planned economic collapse will trigger martial law this year!",
    "BREAKING: Scientists discover genetically modified crops are causing mass sterility!",
    "COVER-UP CONFIRMED: Government hiding alien contact made decades ago from citizens!",
    "DANGER: New legislation covertly strips citizens of all remaining constitutional rights!",
    "ALERT: The education system is deliberately lowering IQ through fluoride and screens!",
    "TRUTH EXPOSED: Weather modification programs are causing all major natural disasters!",
    "ACT IMMEDIATELY: This blood-thinning agent is secretly added to all bottled water!",
    "BOMBSHELL: Leaked Pentagon documents confirm plan to stage a nuclear false flag event!",
    "FINAL PROOF: Elections are rigged by quantum supercomputers controlled by globalists!",
    "BREAKING ALERT: The food crisis is manufactured — food is being burned by the elite!",
    "URGENT: Children being secretly microchipped through routine school vaccination programs!",
    "EXPOSED COVER-UP: Thousands dying from 5G radiation but mainstream media is silent!",
    "WARNING: New digital ID system will give the government total control over your life!",
    "SHOCKING: Secret society meetings reveal plan to collapse global financial system!",
    "THEY ARE SPRAYING: Aerial chemical sprays are triggering engineered respiratory illness!",
    "CRITICAL ALERT: Scientists fired for publishing data that contradicts official narrative!",
    "BREAKING NEWS SUPPRESSED: Entire city water system contaminated with neurotoxins instantly!",
    "END TIMES WARNING: Biblical prophecy and scientific data both point to imminent disaster!",
    "SHARE NOW BEFORE REMOVAL: This whistleblower exposes global genocide in progress!",
    "URGENT FINAL CALL: Time is running out — join the resistance before total global collapse!",
    "ACT NOW OR NEVER: The last free election is about to be permanently stolen by the elite!",
    "EXPOSED: Chemotherapy intentionally kept ineffective so hospitals profit from dying patients!",
    "PANIC-WORTHY: 90% of financial experts privately say a crash is coming this weekend!",
    "BREAKING: Satellite data confirms government is manipulating wind patterns for control!",
    "URGENT BRIEFING: Insider reveals foreign troops pre-positioned for domestic occupation!",
    "LAST WARNING: Digital currency rollout will eliminate financial freedom permanently!",
    "TRUTH LEAK: Scientists confirm that mainstream news is scripted by a single corporation!",
    "BOMBSHELL REVELATION: The vaccines contain biological warfare agents, documents show!",
]

# ── Assemble and balance dataset ──────────────────────────────────────────────
def build_dataset():
    random.shuffle(LOW_RISK_SAMPLES)
    random.shuffle(MODERATE_RISK_SAMPLES)
    random.shuffle(HIGH_RISK_SAMPLES)

    # Pick 67, 67, 66 → 200 samples total
    samples = (
        [(text, "Low") for text in LOW_RISK_SAMPLES[:67]] +
        [(text, "Moderate") for text in MODERATE_RISK_SAMPLES[:67]] +
        [(text, "High") for text in HIGH_RISK_SAMPLES[:66]]
    )

    random.shuffle(samples)
    return samples


if __name__ == "__main__":
    dataset = build_dataset()

    output_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(dataset)

    print(f"✅ Dataset saved → {output_path}")
    print(f"   Total samples : {len(dataset)}")
    label_counts = {}
    for _, lbl in dataset:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    for lbl, count in sorted(label_counts.items()):
        print(f"   {lbl:10s}: {count}")
