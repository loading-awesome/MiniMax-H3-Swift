import Foundation

/// What this Mac is, measured at startup rather than assumed.
///
/// Every field here comes from `sysctl` or `host_statistics64`. Nothing is
/// inferred from a marketing name, because the same marketing name ships with
/// memory configurations that differ by a factor of eight — an "M3 Ultra Mac
/// Studio" is 96, 256 or 512 GB, and only one of those runs this model in bf16.
public struct Machine: Sendable, Equatable {

    /// `hw.model`, e.g. `Mac16,9`.
    public let model: String
    /// `machdep.cpu.brand_string`, e.g. `Apple M3 Ultra`.
    public let chip: String
    /// Physical memory in bytes, from `hw.memsize`.
    public let memoryBytes: UInt64
    /// Performance + efficiency cores.
    public let cores: Int
    /// True when the enclosure is a notebook.
    ///
    /// This is not cosmetic. The throughput this library estimates from was
    /// measured on a desktop under sustained load; a notebook throttles over a
    /// twenty-minute render and will finish late. The estimate says so rather
    /// than being quietly wrong.
    public let isPortable: Bool

    public var memoryGB: Double { Double(memoryBytes) / 1e9 }

    // MARK: measurement

    public static func detect() -> Machine {
        Machine(model: sysctlString("hw.model") ?? "unknown",
                chip: sysctlString("machdep.cpu.brand_string") ?? "unknown",
                memoryBytes: sysctlUInt64("hw.memsize") ?? 0,
                cores: Int(sysctlUInt64("hw.ncpu") ?? 0),
                isPortable: (sysctlString("hw.model") ?? "").hasPrefix("Mac")
                            && (sysctlString("hw.model") ?? "").contains("Book")
                            || (sysctlString("hw.model") ?? "").hasPrefix("MacBook"))
    }

    /// Memory not currently spoken for, in bytes.
    ///
    /// **A 64 GB machine with a browser open is not a 64 GB machine**, and the
    /// difference is not academic: a render was SIGKILLed at 196 GB on a 275 GB
    /// box because a second renderer was running. Planning against `hw.memsize`
    /// would have called that configuration comfortable.
    ///
    /// Free plus inactive plus speculative, all of which the kernel can hand
    /// over under pressure. Wired and active pages cannot be counted on.
    public static func availableBytes() -> UInt64 {
        var size = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size
                                          / MemoryLayout<integer_t>.size)
        var stats = vm_statistics64_data_t()
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(size)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &size)
            }
        }
        guard result == KERN_SUCCESS else { return sysctlUInt64("hw.memsize") ?? 0 }
        // `vm_kernel_page_size` is a mutable global and not concurrency-safe
        // to read under strict checking. `hw.pagesize` is the same number from
        // an interface that is.
        let page = sysctlUInt64("hw.pagesize") ?? 16384
        return (UInt64(stats.free_count) + UInt64(stats.inactive_count)
                + UInt64(stats.speculative_count)) * page
    }

    // MARK: sysctl helpers

    static func sysctlString(_ name: String) -> String? {
        var len = 0
        guard sysctlbyname(name, nil, &len, nil, 0) == 0, len > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: len)
        guard sysctlbyname(name, &buf, &len, nil, 0) == 0 else { return nil }
        return String(cString: buf)
    }

    static func sysctlUInt64(_ name: String) -> UInt64? {
        var value: UInt64 = 0
        var len = MemoryLayout<UInt64>.size
        guard sysctlbyname(name, &value, &len, nil, 0) == 0 else { return nil }
        return value
    }

    public var summary: String {
        String(format: "%@ (%@), %.0f GB unified memory, %d cores%@",
               chip, model, memoryGB, cores, isPortable ? ", portable" : "")
    }
}
