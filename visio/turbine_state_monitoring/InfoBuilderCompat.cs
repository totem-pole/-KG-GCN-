using OfficeIMO.Visio.Fluent;

internal static class InfoBuilderCompatExtensions {
    // OfficeIMO.Visio currently exposes Title/Author only. Keep the generator
    // source readable while treating Subject as optional metadata.
    public static InfoBuilder Subject(this InfoBuilder builder, string subject) => builder;
}
