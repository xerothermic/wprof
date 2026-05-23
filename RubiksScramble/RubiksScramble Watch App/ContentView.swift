import SwiftUI

struct ContentView: View {
    @State private var length = ScrambleGenerator.defaultLength
    @State private var moves = ScrambleGenerator.generate()

    var body: some View {
        TabView {
            ScrambleScreen(length: $length, moves: $moves)
            CubeMapView(moves: moves)
        }
        .tabViewStyle(.verticalPage)
    }
}

private struct ScrambleScreen: View {
    @Binding var length: Int
    @Binding var moves: [String]

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

                    Button {
                        moves = ScrambleGenerator.generate(length: length)
                    } label: {
                        Label("New Scramble", systemImage: "shuffle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)

                    VStack(spacing: 2) {
                        Image(systemName: "chevron.compact.down")
                        Text("Cube map")
                            .font(.caption2)
                    }
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
                }
                .padding(.horizontal, 4)
            }
            .navigationTitle("Scramble")
        }
    }
}

#Preview {
    ContentView()
}
