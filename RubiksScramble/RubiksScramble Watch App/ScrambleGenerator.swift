import Foundation

/// The six faces of a 3x3 cube in standard WCA notation.
enum CubeFace: String, CaseIterable {
    case up = "U"
    case down = "D"
    case left = "L"
    case right = "R"
    case front = "F"
    case back = "B"
}

/// Generates random 3x3 Rubik's cube scrambles in standard notation.
///
/// Rules:
///   1. Default length is 20 moves (configurable).
///   2. Standard outer-face notation only — no slice (M/E/S) or wide moves.
///   3. The same face never turns twice in a row.
enum ScrambleGenerator {
    static let defaultLength = 20

    /// Each face turn carries one modifier: "" (90° CW), "'" (90° CCW), or "2" (180°).
    private static let modifiers = ["", "'", "2"]

    static func generate(length: Int = defaultLength) -> [String] {
        var moves: [String] = []
        moves.reserveCapacity(max(0, length))
        var lastFace: CubeFace?

        for _ in 0..<max(0, length) {
            var face = CubeFace.allCases.randomElement()!
            while face == lastFace {
                face = CubeFace.allCases.randomElement()!
            }
            lastFace = face
            moves.append(face.rawValue + modifiers.randomElement()!)
        }
        return moves
    }

    static func generateString(length: Int = defaultLength) -> String {
        generate(length: length).joined(separator: " ")
    }
}
