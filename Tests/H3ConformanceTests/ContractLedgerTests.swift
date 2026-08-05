import Testing
@testable import H3Conformance

@Suite("fragile contract ledger")
struct ContractLedgerTests {
    @Test("all 36 contracts have exactly one accountable entry")
    func complete() {
        let ids = H3Conformance.contracts.map(\.id)
        #expect(ids.sorted() == Array(1 ... 36))
        #expect(Set(ids).count == ids.count)
    }

    @Test("every contract names an owner, evidence, tier, and mechanism")
    func accountable() {
        for contract in H3Conformance.contracts {
            #expect(!contract.title.isEmpty)
            #expect(!contract.owner.isEmpty)
            #expect(contract.evidence == "FRAGILE_CONTRACTS.md #\(contract.id)")
        }
        #expect(Set(H3Conformance.contracts.map(\.tier)) == Set(H3Conformance.Tier.allCases))
    }

    @Test("manual quality decisions are confined to the release tier")
    func qualityGatesAreReleaseWork() {
        let quality = H3Conformance.contracts.filter { $0.mechanism == .releaseQualityGate }
        #expect(!quality.isEmpty)
        #expect(quality.allSatisfy { $0.tier == .release })
    }
}
