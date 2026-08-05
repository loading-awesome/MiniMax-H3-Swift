import Foundation

// The retained numerical checks.
//
// The CUDA capture tooling, the golden bundles and the RunPod scripts were
// discovery and are gone. These are not: this codebase's failure mode is
// specifically silent — a wrong packed layout, a dropped `<Audio>` label, a
// transposed qkv all keep every tensor the right shape — so the defence against
// regression has to be numerical, and it has to be cheap enough to run on every
// commit. Kilobytes of recorded tensors, no GPU, seconds to run.
//
// Each check corresponds to an entry in FRAGILE_CONTRACTS.md. A contract with
// no check here is a contract that is written down but not enforced.
public enum H3Conformance {
    public static let contractsCovered: [Int] = []
}
