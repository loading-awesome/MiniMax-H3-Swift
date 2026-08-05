import Foundation
import H3Foundation

// Phase-scoped residency lives here. The ordering below is a memory contract,
// not a style choice: encoding every condition *before* the DiT is loaded is
// what keeps the 51.5 GB text encoder and the 66.3 GB DiT from being resident
// at the same time.
//
//   1. text conditioning   -> release, clear cache
//   2. VAE encode          -> release, clear cache
//   3. sample              -> keep the cache; the sampler wants it
//   4. decode + mux
public enum H3Pipeline {
    public static let phaseOrder = ["textEncode", "vaeEncode", "sampling", "decode"]
}
