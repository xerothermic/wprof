import SwiftUI

struct ContentView: View {
    @State private var length = ScrambleGenerator.defaultLength
    @State private var moves = ScrambleGenerator.generate()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 12) {
                    Text(moves.joined(separator: "  "))
                        .font(.system(.headline, design: .monospaced))
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 4)

                    Stepper(value: $length, in: 1...50) {
                        Text("Length: \(length)")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    Button(action: newScramble) {
                        Label("New Scramble", systemImage: "shuffle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding(.horizontal, 4)
            }
            .navigationTitle("Scramble")
        }
    }

    private func newScramble() {
        moves = ScrambleGenerator.generate(length: length)
    }
}

#Preview {
    ContentView()
}
