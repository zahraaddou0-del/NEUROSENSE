# ══════════════════════════════════════════════════════════════════
# عرض الصفحة الأولى فوراً (بدون انتظار التدريب)
# ══════════════════════════════════════════════════════════════════

# أولاً، نعرض الصفحة الحالية حسب رقمها
if st.session_state.page == 1:
    # عرض صفحة اختيار الدور مباشرة (بدون انتظار)
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("👋 Qui êtes-vous ?")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="text-align:center;background:linear-gradient(135deg,#667eea20,#764ba220);
                    border-radius:20px;padding:2rem;">
            <span style="font-size:4rem;">👨‍👩‍👧</span>
            <h3>Parent</h3>
            <p>Complétez le questionnaire pour votre enfant</p>
        </div>""", unsafe_allow_html=True)
        if st.button("📝 Je suis un parent", key="btn_parent", use_container_width=True):
            st.session_state.role = "parent"
            st.session_state.page = 2
            st.rerun()
    with c2:
        st.markdown("""
        <div style="text-align:center;background:linear-gradient(135deg,#667eea20,#764ba220);
                    border-radius:20px;padding:2rem;">
            <span style="font-size:4rem;">👨‍⚕️</span>
            <h3>Médecin</h3>
            <p>Évaluez votre patient avec notre outil d'aide</p>
        </div>""", unsafe_allow_html=True)
        if st.button("🩺 Je suis médecin", key="btn_medecin", use_container_width=True):
            st.session_state.role = "medecin"
            st.session_state.page = 2
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ثم نقوم بتحميل النماذج في الخلفية (إذا لم تكن محملة بالفعل)
elif st.session_state.page > 1 and not st.session_state.model_entraine:
    # فقط عندما يحتاج المستخدم النماذج، نبدأ التدريب
    with st.spinner("🤖 Chargement des modèles d'IA... (30-60 secondes)"):
        if st.session_state.df_train is None:
            df_raw, _ = charger_donnees()
            st.session_state.df_train = df_raw
        
        (model, best_name, all_results, le_dict, scaler,
         X_tr, X_te, y_tr, y_te, y_pred, accuracy, col_names) = \
            pipeline_complet(st.session_state.df_train)

        st.session_state.update(dict(
            model=model, best_name=best_name, all_results=all_results,
            le_dict=le_dict, scaler=scaler, X_train=X_tr, X_test=X_te,
            y_train=y_tr, y_test=y_te, y_pred=y_pred, accuracy=accuracy,
            col_names=col_names, model_entraine=True,
        ))
    st.rerun()
