<View>
  <Style>
    /* ---------- Base ---------- */
    .ls-root {
      font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* ---------- Paper header ---------- */
    .paper-header {
      background: linear-gradient(135deg, #eef2ff 0%, #f5f8ff 100%);
      border: 1px solid #dbe3fb;
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 22px;
      box-shadow: 0 1px 3px rgba(30, 41, 90, 0.06);
    }
    .paper-header h1, .paper-header h2, .paper-header h3, .paper-header h4 {
      color: #1e2a5e !important;
      margin-top: 0 !important;
    }
    .part-badge {
      display: inline-block;
      margin-top: 6px;
      padding: 2px 10px;
      border-radius: 999px;
      background: #eef2ff;
      color: #4338ca !important;
      font-size: 12px;
      font-weight: 700;
    }

    /* ---------- Section title ---------- */
    .section-title {
      color: #1e2a5e !important;
      border-bottom: 2px solid #dbe3fb;
      padding-bottom: 8px;
      margin-top: 26px !important;
      margin-bottom: 14px !important;
    }

    /* ---------- Record card ---------- */
    .record-block {
      background: #ffffff;
      border: 1px solid #e6e9f2;
      border-left: 5px solid #4338ca;
      border-radius: 12px;
      padding: 20px 22px;
      margin-bottom: 22px;
      box-shadow: 0 1px 4px rgba(30, 41, 90, 0.06);
    }
    .record-title { color: #4338ca !important; margin-top: 0 !important; margin-bottom: 4px !important; }

    /* ---------- Field row ---------- */
    .field-row { margin-bottom: 16px; }

    .field-badge {
      display: inline-block;
      padding: 3px 11px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: #ffffff !important;
      margin-bottom: 6px;
    }
    .field-badge span, .field-badge div { color: #ffffff !important; }
    .badge-term         { background: #4338ca; }
    .badge-definitions  { background: #2563eb; }
    .badge-synonyms     { background: #0d9488; }
    .badge-description  { background: #64748b; }
    .badge-related      { background: #d97706; }
    .badge-subtypes     { background: #db2777; }
    .badge-supertypes   { background: #7c3aed; }
    .badge-examples     { background: #16a34a; }

    .field-value {
      background: #f8f9fc;
      border: 1px solid #e6e9f2;
      border-radius: 8px;
      padding: 10px 12px;
      margin: 0 0 8px 0;
      white-space: pre-wrap;
      color: #2b3350;
      font-size: 14px;
      line-height: 1.45;
    }

    /* ---------- Term relevance question (drives collapse) ---------- */
    .term-gate {
      background: #f5f3ff;
      border: 1px solid #ddd6fe;
      border-radius: 10px;
      padding: 12px 14px;
      margin-top: 6px;
    }

    /* ---------- "Is a definition?" check ---------- */
    .definition-check {
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 10px;
      padding: 10px 14px;
      margin-top: 4px;
      margin-bottom: 8px;
    }

    /* ---------- Not-relevant collapsed notice ---------- */
    .not-relevant-notice {
      background: #fef2f2;
      border: 1px dashed #fca5a5;
      border-radius: 10px;
      padding: 14px 16px;
      color: #991b1b;
      font-weight: 600;
      margin-top: 10px;
    }

    /* ---------- Error flag (typo etc.) ---------- */
    .error-flag-row { margin-top: 6px; }
    .error-flag-row .ant-checkbox-wrapper,
    .error-flag-row label { color: #b45309 !important; }
    .error-textarea textarea {
      background: #fffbeb !important;
      border: 1px solid #fcd34d !important;
      border-radius: 6px !important;
    }

    /* ---------- Example rows ---------- */
    .example-row {
      background: #f0fdf4;
      border: 1px solid #dcfce7;
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }
    .example-row .field-value { background: #ffffff; margin-bottom: 8px; }
  </Style>

  <View className="ls-root">

    <!-- ============ Paper header (no PDF) ============ -->
    <View className="paper-header">
      <Header value="$title" size="3"/>
      <Text name="meta_authors" value="Authors: $authors_display"/>
      <Text name="meta_info" value="$meta_display"/>
      <Text name="meta_ids" value="Paper ID: $original_paper_id   |   Split: $split"/>
      <View className="part-badge"><Text name="meta_part" value="$part_display"/></View>
    </View>

    <Header value="Definition Records" size="4" className="section-title"/>

    <!-- ============ One block per definition_record ============ -->
    <Repeater on="$definition_records" indexFlag="{{idx}}">
      <View className="record-block">
        <Header value="Concept: $definition_records[{{idx}}].term" size="4" className="record-title"/>

        <!-- ===== TERM + DEFINITION + relevance gate ===== -->
        <View className="field-row">
          <View className="field-badge badge-term"><Text name="term_label_{{idx}}" value="TERM"/></View>
          <View className="field-value"><Text name="term_value_{{idx}}" value="$definition_records[{{idx}}].term"/></View>

          <View className="field-badge badge-definitions"><Text name="definitions_label_{{idx}}" value="DEFINITIONS"/></View>
          <View className="field-value"><Text name="definitions_value_{{idx}}" value="$definition_records[{{idx}}].definitions_display"/></View>

          <View className="definition-check">
            <Text name="definition_is_definition_label_{{idx}}" value="Is a definition?"/>
            <Choices name="definition_is_definition_{{idx}}" toName="definition_is_definition_label_{{idx}}" choice="single" showInline="true">
              <Choice value="Yes" style="accent-color:#16a34a;"/>
              <Choice value="No" style="accent-color:#dc2626;"/>
            </Choices>
          </View>

          <View className="error-flag-row">
            <Choices name="term_definition_error_{{idx}}" toName="definitions_value_{{idx}}" choice="multiple" showInline="true">
              <Choice value="Contains an error (e.g. typo)" style="accent-color:#b45309;"/>
            </Choices>
          </View>
          <View visibleWhen="choice-selected" whenTagName="term_definition_error_{{idx}}" whenChoiceValue="Contains an error (e.g. typo)" className="error-textarea">
            <TextArea name="term_definition_error_text_{{idx}}" toName="definitions_value_{{idx}}"
                      placeholder="Optional: describe the error" rows="2" editable="true" maxSubmissions="1"/>
          </View>

          <View className="term-gate">
            <Text name="term_relevant_label_{{idx}}" value="Is the concept relevant for the Media Bias Taxonomy task?"/>
            <Choices name="term_relevant_{{idx}}" toName="term_relevant_label_{{idx}}" choice="single" required="true" showInline="true">
              <Choice value="Yes" style="accent-color:#16a34a;"/>
              <Choice value="No" style="accent-color:#dc2626;"/>
            </Choices>
          </View>
        </View>

        <!-- ===== Everything below only if the concept IS relevant ===== -->
        <View visibleWhen="choice-selected" whenTagName="term_relevant_{{idx}}" whenChoiceValue="Yes">

          <!-- SYNONYMS -->
          <View className="field-row">
            <View className="field-badge badge-synonyms"><Text name="synonyms_label_{{idx}}" value="SYNONYMS"/></View>
            <View className="field-value"><Text name="synonyms_value_{{idx}}" value="$definition_records[{{idx}}].synonyms_display"/></View>
            <Choices name="synonyms_fit_{{idx}}" toName="synonyms_value_{{idx}}" choice="single" showInline="true">
              <Choice value="Does fit" style="accent-color:#0d9488;"/>
              <Choice value="Does not fit" style="accent-color:#0d9488;"/>
            </Choices>
            <View className="error-flag-row">
              <Choices name="synonyms_error_{{idx}}" toName="synonyms_value_{{idx}}" choice="multiple" showInline="true">
                <Choice value="Contains an error (e.g. typo)" style="accent-color:#b45309;"/>
              </Choices>
            </View>
            <View visibleWhen="choice-selected" whenTagName="synonyms_error_{{idx}}" whenChoiceValue="Contains an error (e.g. typo)" className="error-textarea">
              <TextArea name="synonyms_error_text_{{idx}}" toName="synonyms_value_{{idx}}"
                        placeholder="Optional: describe the error" rows="2" editable="true" maxSubmissions="1"/>
            </View>
          </View>

          <!-- RELATED TERMS -->
          <View className="field-row">
            <View className="field-badge badge-related"><Text name="related_terms_label_{{idx}}" value="RELATED TERMS"/></View>
            <View className="field-value"><Text name="related_terms_value_{{idx}}" value="$definition_records[{{idx}}].related_terms_display"/></View>
            <Choices name="related_terms_fit_{{idx}}" toName="related_terms_value_{{idx}}" choice="single" showInline="true">
              <Choice value="Does fit" style="accent-color:#d97706;"/>
              <Choice value="Does not fit" style="accent-color:#d97706;"/>
            </Choices>
            <View className="error-flag-row">
              <Choices name="related_terms_error_{{idx}}" toName="related_terms_value_{{idx}}" choice="multiple" showInline="true">
                <Choice value="Contains an error (e.g. typo)" style="accent-color:#b45309;"/>
              </Choices>
            </View>
            <View visibleWhen="choice-selected" whenTagName="related_terms_error_{{idx}}" whenChoiceValue="Contains an error (e.g. typo)" className="error-textarea">
              <TextArea name="related_terms_error_text_{{idx}}" toName="related_terms_value_{{idx}}"
                        placeholder="Optional: describe the error" rows="2" editable="true" maxSubmissions="1"/>
            </View>
          </View>

          <!-- DESCRIPTION -->
          <View className="field-row">
            <View className="field-badge badge-description"><Text name="description_label_{{idx}}" value="DESCRIPTION"/></View>
            <View className="field-value"><Text name="description_value_{{idx}}" value="$definition_records[{{idx}}].description_display"/></View>
            <Choices name="description_fit_{{idx}}" toName="description_value_{{idx}}" choice="single" showInline="true">
              <Choice value="Does fit" style="accent-color:#64748b;"/>
              <Choice value="Does not fit" style="accent-color:#64748b;"/>
            </Choices>
            <View className="error-flag-row">
              <Choices name="description_error_{{idx}}" toName="description_value_{{idx}}" choice="multiple" showInline="true">
                <Choice value="Contains an error (e.g. typo)" style="accent-color:#b45309;"/>
              </Choices>
            </View>
            <View visibleWhen="choice-selected" whenTagName="description_error_{{idx}}" whenChoiceValue="Contains an error (e.g. typo)" className="error-textarea">
              <TextArea name="description_error_text_{{idx}}" toName="description_value_{{idx}}"
                        placeholder="Optional: describe the error" rows="2" editable="true" maxSubmissions="1"/>
            </View>
          </View>

          <!-- SUBTYPES -->
          <View className="field-row">
            <View className="field-badge badge-subtypes"><Text name="subtypes_label_{{idx}}" value="SUBTYPES"/></View>
            <View className="field-value"><Text name="subtypes_value_{{idx}}" value="$definition_records[{{idx}}].subtypes_display"/></View>
            <Choices name="subtypes_fit_{{idx}}" toName="subtypes_value_{{idx}}" choice="single" showInline="true">
              <Choice value="Does fit" style="accent-color:#db2777;"/>
              <Choice value="Does not fit" style="accent-color:#db2777;"/>
            </Choices>
            <View className="error-flag-row">
              <Choices name="subtypes_error_{{idx}}" toName="subtypes_value_{{idx}}" choice="multiple" showInline="true">
                <Choice value="Contains an error (e.g. typo)" style="accent-color:#b45309;"/>
              </Choices>
            </View>
            <View visibleWhen="choice-selected" whenTagName="subtypes_error_{{idx}}" whenChoiceValue="Contains an error (e.g. typo)" className="error-textarea">
              <TextArea name="subtypes_error_text_{{idx}}" toName="subtypes_value_{{idx}}"
                        placeholder="Optional: describe the error" rows="2" editable="true" maxSubmissions="1"/>
            </View>
          </View>

          <!-- SUPERTYPES -->
          <View className="field-row">
            <View className="field-badge badge-supertypes"><Text name="supertypes_label_{{idx}}" value="SUPERTYPES"/></View>
            <View className="field-value"><Text name="supertypes_value_{{idx}}" value="$definition_records[{{idx}}].supertypes_display"/></View>
            <Choices name="supertypes_fit_{{idx}}" toName="supertypes_value_{{idx}}" choice="single" showInline="true">
              <Choice value="Does fit" style="accent-color:#7c3aed;"/>
              <Choice value="Does not fit" style="accent-color:#7c3aed;"/>
            </Choices>
            <View className="error-flag-row">
              <Choices name="supertypes_error_{{idx}}" toName="supertypes_value_{{idx}}" choice="multiple" showInline="true">
                <Choice value="Contains an error (e.g. typo)" style="accent-color:#b45309;"/>
              </Choices>
            </View>
            <View visibleWhen="choice-selected" whenTagName="supertypes_error_{{idx}}" whenChoiceValue="Contains an error (e.g. typo)" className="error-textarea">
              <TextArea name="supertypes_error_text_{{idx}}" toName="supertypes_value_{{idx}}"
                        placeholder="Optional: describe the error" rows="2" editable="true" maxSubmissions="1"/>
            </View>
          </View>

          <!-- EXAMPLES: listed individually, each with type + fit + error flag -->
          <View className="field-row">
            <View className="field-badge badge-examples"><Text name="examples_label_{{idx}}" value="EXAMPLES"/></View>

            <View className="example-row">
              <View className="field-value"><Text name="example_{{idx}}_0_value" value="$definition_records[{{idx}}].examples[0]"/></View>

              <Choices name="example_{{idx}}_0_fit" toName="example_{{idx}}_0_value" choice="single" showInline="true">
                <Choice value="Does fit" style="accent-color:#0891b2;"/>
                <Choice value="Does not fit" style="accent-color:#0891b2;"/>
              </Choices>

              <View visibleWhen="choice-selected" whenTagName="example_{{idx}}_0_fit" whenChoiceValue="Does fit">
                <Choices name="example_{{idx}}_0_type" toName="example_{{idx}}_0_value" choice="single" showInline="true">
                  <Choice value="Example" style="accent-color:#16a34a;"/>
                  <Choice value="Counterexample" style="accent-color:#dc2626;"/>
                  <Choice value="Others" style="accent-color:#9ca3af;"/>
                </Choices>
              </View>

              <View className="error-flag-row">
                <Choices name="example_{{idx}}_0_error" toName="example_{{idx}}_0_value" choice="multiple" showInline="true">
                  <Choice value="Contains an error (e.g. typo)" style="accent-color:#b45309;"/>
                </Choices>
              </View>
              <View visibleWhen="choice-selected" whenTagName="example_{{idx}}_0_error" whenChoiceValue="Contains an error (e.g. typo)" className="error-textarea">
                <TextArea name="example_{{idx}}_0_error_text" toName="example_{{idx}}_0_value"
                          placeholder="Optional: describe the error" rows="2" editable="true" maxSubmissions="1"/>
              </View>
            </View>

            <View className="example-row">
              <View className="field-value"><Text name="example_{{idx}}_1_value" value="$definition_records[{{idx}}].examples[1]"/></View>

              <Choices name="example_{{idx}}_1_fit" toName="example_{{idx}}_1_value" choice="single" showInline="true">
                <Choice value="Does fit" style="accent-color:#0891b2;"/>
                <Choice value="Does not fit" style="accent-color:#0891b2;"/>
              </Choices>

              <View visibleWhen="choice-selected" whenTagName="example_{{idx}}_1_fit" whenChoiceValue="Does fit">
                <Choices name="example_{{idx}}_1_type" toName="example_{{idx}}_1_value" choice="single" showInline="true">
                  <Choice value="Example" style="accent-color:#16a34a;"/>
                  <Choice value="Counterexample" style="accent-color:#dc2626;"/>
                  <Choice value="Others" style="accent-color:#9ca3af;"/>
                </Choices>
              </View>

              <View className="error-flag-row">
                <Choices name="example_{{idx}}_1_error" toName="example_{{idx}}_1_value" choice="multiple" showInline="true">
                  <Choice value="Contains an error (e.g. typo)" style="accent-color:#b45309;"/>
                </Choices>
              </View>
              <View visibleWhen="choice-selected" whenTagName="example_{{idx}}_1_error" whenChoiceValue="Contains an error (e.g. typo)" className="error-textarea">
                <TextArea name="example_{{idx}}_1_error_text" toName="example_{{idx}}_1_value"
                          placeholder="Optional: describe the error" rows="2" editable="true" maxSubmissions="1"/>
              </View>
            </View>

            <View className="example-row">
              <View className="field-value"><Text name="example_{{idx}}_2_value" value="$definition_records[{{idx}}].examples[2]"/></View>

              <Choices name="example_{{idx}}_2_fit" toName="example_{{idx}}_2_value" choice="single" showInline="true">
                <Choice value="Does fit" style="accent-color:#0891b2;"/>
                <Choice value="Does not fit" style="accent-color:#0891b2;"/>
              </Choices>

              <View visibleWhen="choice-selected" whenTagName="example_{{idx}}_2_fit" whenChoiceValue="Does fit">
                <Choices name="example_{{idx}}_2_type" toName="example_{{idx}}_2_value" choice="single" showInline="true">
                  <Choice value="Example" style="accent-color:#16a34a;"/>
                  <Choice value="Counterexample" style="accent-color:#dc2626;"/>
                  <Choice value="Others" style="accent-color:#9ca3af;"/>
                </Choices>
              </View>

              <View className="error-flag-row">
                <Choices name="example_{{idx}}_2_error" toName="example_{{idx}}_2_value" choice="multiple" showInline="true">
                  <Choice value="Contains an error (e.g. typo)" style="accent-color:#b45309;"/>
                </Choices>
              </View>
              <View visibleWhen="choice-selected" whenTagName="example_{{idx}}_2_error" whenChoiceValue="Contains an error (e.g. typo)" className="error-textarea">
                <TextArea name="example_{{idx}}_2_error_text" toName="example_{{idx}}_2_value"
                          placeholder="Optional: describe the error" rows="2" editable="true" maxSubmissions="1"/>
              </View>
            </View>

            <View className="example-row">
              <View className="field-value"><Text name="example_{{idx}}_3_value" value="$definition_records[{{idx}}].examples[3]"/></View>

              <Choices name="example_{{idx}}_3_fit" toName="example_{{idx}}_3_value" choice="single" showInline="true">
                <Choice value="Does fit" style="accent-color:#0891b2;"/>
                <Choice value="Does not fit" style="accent-color:#0891b2;"/>
              </Choices>

              <View visibleWhen="choice-selected" whenTagName="example_{{idx}}_3_fit" whenChoiceValue="Does fit">
                <Choices name="example_{{idx}}_3_type" toName="example_{{idx}}_3_value" choice="single" showInline="true">
                  <Choice value="Example" style="accent-color:#16a34a;"/>
                  <Choice value="Counterexample" style="accent-color:#dc2626;"/>
                  <Choice value="Others" style="accent-color:#9ca3af;"/>
                </Choices>
              </View>

              <View className="error-flag-row">
                <Choices name="example_{{idx}}_3_error" toName="example_{{idx}}_3_value" choice="multiple" showInline="true">
                  <Choice value="Contains an error (e.g. typo)" style="accent-color:#b45309;"/>
                </Choices>
              </View>
              <View visibleWhen="choice-selected" whenTagName="example_{{idx}}_3_error" whenChoiceValue="Contains an error (e.g. typo)" className="error-textarea">
                <TextArea name="example_{{idx}}_3_error_text" toName="example_{{idx}}_3_value"
                          placeholder="Optional: describe the error" rows="2" editable="true" maxSubmissions="1"/>
              </View>
            </View>

            <View className="example-row">
              <View className="field-value"><Text name="example_{{idx}}_4_value" value="$definition_records[{{idx}}].examples[4]"/></View>

              <Choices name="example_{{idx}}_4_fit" toName="example_{{idx}}_4_value" choice="single" showInline="true">
                <Choice value="Does fit" style="accent-color:#0891b2;"/>
                <Choice value="Does not fit" style="accent-color:#0891b2;"/>
              </Choices>

              <View visibleWhen="choice-selected" whenTagName="example_{{idx}}_4_fit" whenChoiceValue="Does fit">
                <Choices name="example_{{idx}}_4_type" toName="example_{{idx}}_4_value" choice="single" showInline="true">
                  <Choice value="Example" style="accent-color:#16a34a;"/>
                  <Choice value="Counterexample" style="accent-color:#dc2626;"/>
                  <Choice value="Others" style="accent-color:#9ca3af;"/>
                </Choices>
              </View>

              <View className="error-flag-row">
                <Choices name="example_{{idx}}_4_error" toName="example_{{idx}}_4_value" choice="multiple" showInline="true">
                  <Choice value="Contains an error (e.g. typo)" style="accent-color:#b45309;"/>
                </Choices>
              </View>
              <View visibleWhen="choice-selected" whenTagName="example_{{idx}}_4_error" whenChoiceValue="Contains an error (e.g. typo)" className="error-textarea">
                <TextArea name="example_{{idx}}_4_error_text" toName="example_{{idx}}_4_value"
                          placeholder="Optional: describe the error" rows="2" editable="true" maxSubmissions="1"/>
              </View>
            </View>

          </View>

        </View>

        <!-- ===== Collapsed notice when the concept is NOT relevant ===== -->
        <View visibleWhen="choice-selected" whenTagName="term_relevant_{{idx}}" whenChoiceValue="No">
          <View className="not-relevant-notice">
            Concept marked as not relevant — this concept card is treated as not usable for the Media Bias Taxonomy task. No further annotation needed here.
          </View>
        </View>

      </View>
    </Repeater>

  </View>
</View>